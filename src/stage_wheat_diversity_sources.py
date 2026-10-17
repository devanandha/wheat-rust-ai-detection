"""Stage verified IARI and Ethiopian wheat-leaf sources for a diversity ablation.

Requires audit_ethiopia_groups.py's image_inventory.csv. Does not modify source
images. Keeps the Ethiopian visually similar image groups together; one representative
per group is used, with excluded members recorded. Uses ordinary file copies by default. Optional hardlinks save space but
share underlying file content; NEVER edit hardlinked staged images in place.

The image-grouping check is not proof of independent plants, fields or sessions.
"""
import argparse
import csv
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
import imagehash

EXTENSIONS = {'.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff'}
CANONICAL = {'control':'Healthy','diseased':'Brown_rust','Healthy':'Healthy','stripe_rust':'Yellow_rust'}

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def rec(path, cls, source, partition):
    with Image.open(path) as im:
        ph = int(str(imagehash.phash(im.convert('RGB'))), 16)
    return {'path':path,'class':cls,'source':source,'partition':partition,'sha':sha(path),'ph':ph}

def image_files(dir):
    if not dir.is_dir():
        raise ValueError(f'Missing required folder: {dir}')
    return [p for p in sorted(dir.rglob('*')) if p.is_file() and p.suffix.lower() in EXTENSIONS]

def copy_ref(r, target_dir, mode):
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{r['source']}_{r['sha'][:16]}_{r['path'].stem}{r['path'].suffix.lower()}"
    if target.exists():
        raise ValueError(f'Staged file already exists: {target}')
    if mode=='hardlink':
        os.link(r['path'],target)
    else:
        import shutil
        shutil.copy2(r['path'],target)
    return str(target)

def similarity(r, other, threshold):
    return r['sha']==other['sha'] or (r['ph'] ^ other['ph']).bit_count()<=threshold

def main(argv=None):
    ap = argparse.ArgumentParser(description='Stage IARI/Ethiopia diversity images without modifying originals')
    ap.add_argument('--iari-root', type=Path, required=True)
    ap.add_argument('--ethiopia-root', type=Path, required=True)
    ap.add_argument('--audit-inventory', type=Path, required=True)
    ap.add_argument('--out-dir',type=Path,required=True)
    ap.add_argument('--split-seed',type=int,default=20261016)
    ap.add_argument('--holdout-fraction',type=float,default=0.20)
    ap.add_argument('--phash-threshold',type=int,default=5)
    ap.add_argument('--mode',choices=['hardlink','copy'],default='copy')
    args=ap.parse_args(argv)
    iari=args.iari_root.resolve(); ethiopia=args.ethiopia_root.resolve(); out=args.out_dir.resolve()
    if not 0.1 <= args.holdout_fraction<=0.4:
        ap.error('--holdout-fraction should be between 0.1 and 0.4')
    if not 0<=args.phash_threshold<=10: ap.error('Choose a pHash threshold from 0 to 10')
    if out.exists(): ap.error(f'Output already exists; use a new destination: {out}')
    if out == iari or iari in out.parents or out==ethiopia or ethiopia in out.parents:
        ap.error('The output directory must be OUTSIDE both dataset directories')
    assert args.audit_inventory.is_file(), 'Ethiopian audit inventory CSV not found'
    with args.audit_inventory.open(encoding='utf-8',newline='') as f:
        inventory=list(csv.DictReader(f))
    ethiopia_recs=[]; groups=defaultdict(list)
    for row in inventory:
        cls=row['class']; rel=row['relative_path']; p=ethiopia/Path(rel)
        if not p.is_file(): raise ValueError(f'Audited image is missing: {p}')
        # Septoria remains out of scope, but its audit was already performed.
        if cls=='septoria': continue
        if cls not in ('Healthy','stripe_rust'): raise ValueError(f'Unexpected Ethiopian label: {cls}')
        r=rec(p,CANONICAL[cls],'ethiopia','unassigned')
        if r['sha']!=row['sha256'] or f"{r['ph']:016x}"!=row['phash']:
            raise ValueError(f'Image changed since audit. Re-run audit: {rel}')
        groups[row['candidate_group']].append(r)
        ethiopia_recs.append(r)
    if len(ethiopia_recs)!=310:
        print(f'NOTE: expected 310 Ethiopian Healthy/stripe-rust files, saw {len(ethiopia_recs)}')
    for group_id, rows in groups.items():
        if len({r['class'] for r in rows})>1:
            raise ValueError(f'Cross-class similarity group requires manual review: {group_id}')
    byclass=defaultdict(list); duplicate_members=[]
    for group_id, rows in groups.items():
        rows.sort(key=lambda r:r['path'].name)
        # Keep only one file from each near-duplicate group, preserving the others in place.
        representative=rows[0]; byclass[representative['class']].append(representative)
        for r in rows[1:]:
            duplicate_members.append({'group':group_id,'excluded_original':str(r['path']),
                                      'kept_original':str(representative['path']),
                                      'sha_equal':r['sha']==representative['sha']})
    if not all(byclass[k] for k in ('Healthy','Yellow_rust')):
        raise ValueError('Missing Healthy or Yellow rust classes')
    rng=random.Random(args.split_seed)
    eth_train=[]; eth_holdout=[]
    for cls in ('Healthy','Yellow_rust'):
        pool=sorted(byclass[cls],key=lambda r:r['path'].as_posix());rng.shuffle(pool)
        hold_count=round(len(pool)*args.holdout_fraction)
        for r in pool[:hold_count]: r['partition']='holdout';eth_holdout.append(r)
        for r in pool[hold_count:]: r['partition']='train';eth_train.append(r)
    iari_train=[];iari_protected=[]
    for partition in ('train','val','test'):
        for original_class in ('control','diseased'):
            cls=CANONICAL[original_class]
            for path in image_files(iari/partition/original_class):
                r=rec(path,cls,'iari',partition)
                (iari_train if partition=='train' else iari_protected).append(r)
    # Make sure IARI's original test/validation and Ethiopia's newly reserved holdouts
    # are not represented in the candidate training images by exact/pHash matches.
    protected=iari_protected+eth_holdout
    candidate=iari_train+eth_train
    clean=[]; excluded_holdout=[]
    for r in candidate:
        match=next((p for p in protected if similarity(r,p,args.phash_threshold)),None)
        if match:
            excluded_holdout.append({'source':r['source'],'class':r['class'],'file':str(r['path']),
                                     'matched_partition':f"{match['source']}/{match['partition']}",
                                     'matched_file':str(match['path']),
                                     'exact':r['sha']==match['sha']})
        else: clean.append(r)
    # Drop additional-source duplicates across training sources, documenting exclusions.
    chosen=[]; excluded_train=[]
    for r in sorted(clean,key=lambda x:(x['source'],x['class'],str(x['path']))):
        match=next((p for p in chosen if similarity(r,p,args.phash_threshold)),None)
        if match:
            excluded_train.append({'class':r['class'],'file':str(r['path']),
                                   'matched_file':str(match['path']),
                                   'exact':r['sha']==match['sha']})
        else: chosen.append(r)
    count=Counter((r['source'],r['class']) for r in chosen)
    # 300-image/50% mixture will require 150 additional-source images PER class.
    if count['iari','Brown_rust']<150 or count['ethiopia','Yellow_rust']<150:
        raise ValueError('Not enough filtered Brown or Yellow rust images for n300/mix50; review exclusions and revise experiment size')
    out.mkdir(parents=True,exist_ok=False)
    refs=[]
    for r in chosen:
        dest=copy_ref(r,out/'sources'/r['source']/r['class'],args.mode)
        refs.append({'source':r['source'],'class':r['class'],'original':str(r['path']),
                     'staged':dest,'sha256':r['sha'],'phash':f"{r['ph']:016x}"})
    # Stage Ethiopian holdout for later independent-source evaluation. IARI's
    # existing val/test remain at original paths and are NOT copied or altered.
    reserved=[]
    for r in eth_holdout:
        dest=copy_ref(r,out/'holdouts'/'ethiopia'/r['class'],args.mode)
        reserved.append({'class':r['class'],'original':str(r['path']),'staged':dest,'sha256':r['sha']})
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'split_seed':args.split_seed,
            'holdout_fraction':args.holdout_fraction,'phash_threshold':args.phash_threshold,
            'staging_mode':args.mode,'source_counts':{s:{c:count[s,c] for c in ('Brown_rust','Healthy','Yellow_rust')}
                                                 for s in ('iari','ethiopia')},
            'ethiopia_holdout_counts':dict(Counter(r['class'] for r in eth_holdout)),
            'iari_original_validation':str(iari/'val'),'iari_original_test':str(iari/'test'),
            'duplicate_group_members_excluded':duplicate_members,
            'training_candidates_excluded_due_to_holdout_similarity':excluded_holdout,
            'training_candidates_excluded_due_to_training_similarity':excluded_train,
            'source_file_manifest':refs,'ethiopia_holdout_file_manifest':reserved,
            'limitations':['Image similarity groups do not identify shared plants, fields or acquisition sessions.',
                           'Evaluation on Ethiopian holdout is not necessarily field-independent.',
                           'Cross-source audit against original training/validation and JIC heldout still required before training.']}
    (out/'staging_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'staged_train_images':len(chosen),'source_counts':report['source_counts'],
                      'ethiopia_holdout_counts':report['ethiopia_holdout_counts'],
                      'duplicate_group_members_excluded':len(duplicate_members),
                      'training_exclusions_due_to_holdout_overlap':len(excluded_holdout),
                      'training_exclusions_due_to_source_overlap':len(excluded_train),
                      'reports':str(out/'staging_manifest.json')},indent=2))
    print('No original dataset files modified. (If you chose hardlinks, do not edit staged images in place.)')

if __name__=='__main__':
    main()
