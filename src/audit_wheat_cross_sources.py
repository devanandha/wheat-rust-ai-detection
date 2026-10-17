"""Read-only cross-source image similarity audit for Wheat Rust diversity study.

Compares staged IARI/Ethiopia training images with original train/internal val,
JIC test, and new IARI/Ethiopia holdouts. Also screens the new holdouts
against the original dataset. Does not delete, edit or copy source images.

A pHash distance <= threshold is a *review flag*, not evidence of a duplicate.
Shared plants/fields/photographers cannot be ruled out using image hashes.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import imagehash
from PIL import Image

EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
CLASSES = ('Brown_rust', 'Healthy', 'Yellow_rust')


def image_files(root):
    if not root.is_dir():
        raise FileNotFoundError(f'Missing image directory: {root}')
    return sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in EXT)


def make_record(path, source, label):
    digest = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            digest.update(chunk)
    with Image.open(path) as im:
        p_hash = int(str(imagehash.phash(im.convert('RGB'))), 16)
    return dict(path=str(path.resolve()), source=source, label=label,
                sha256=digest.hexdigest(), phash=p_hash)


def scan(root, source, expected_classes):
    result = []
    for label in expected_classes:
        result += [make_record(p, source, label) for p in image_files(root / label)]
    return result


def add_flag(flags, a, b, threshold, purpose):
    distance = (a['phash'] ^ b['phash']).bit_count()
    identical = a['sha256'] == b['sha256']
    if identical or distance <= threshold:
        flags.append(dict(purpose=purpose,
                          candidate_source=a['source'], candidate_class=a['label'],
                          candidate_path=a['path'],
                          matched_source=b['source'], matched_class=b['label'],
                          matched_path=b['path'], exact_sha256_match=identical,
                          phash_distance=distance, cross_class=(a['label'] != b['label'])))


def compare(left, right, threshold, purpose):
    flags=[]
    for a in left:
        for b in right:
            add_flag(flags, a, b, threshold, purpose)
    return flags


def write_csv(path, rows, cols):
    with path.open('w', newline='', encoding='utf-8') as fp:
        writer=csv.DictWriter(fp, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description='Read-only source overlap audit, outputting candidate review flags')
    p.add_argument('--staging-manifest', type=Path, required=True)
    p.add_argument('--original-train', type=Path, required=True)
    p.add_argument('--original-val', type=Path, required=True)
    p.add_argument('--jic-val', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--phash-threshold', type=int, default=5)
    args=p.parse_args()
    if not (0 <= args.phash_threshold <= 10):
        p.error('phash-threshold must be between 0 and 10')
    manifest=json.loads(args.staging_manifest.read_text(encoding='utf-8'))
    output=args.out_dir.resolve()
    if output.exists(): p.error(f'Output path already exists: {output}; use a new output folder')
    sources=defaultdict(list)
    print('Loading staged training images...', flush=True)
    for row in manifest['source_file_manifest']:
        path=Path(row['staged'])
        if not path.is_file(): raise FileNotFoundError(f'Missing staged image: {path}')
        r=make_record(path, 'staged_'+row['source'],row['class'])
        if r['sha256'] != row['sha256']:
            raise ValueError(f'Staged image hash differs from manifest: {path}')
        sources['staged_train'].append(r)
    print('Loading reserved Ethiopian holdout images...', flush=True)
    for row in manifest['ethiopia_holdout_file_manifest']:
        path=Path(row['staged'])
        if not path.is_file(): raise FileNotFoundError(f'Missing Ethiopian holdout: {path}')
        r=make_record(path,'ethiopia_holdout',row['class'])
        if r['sha256'] != row['sha256']:
            raise ValueError(f'Holdout image hash differs from manifest: {path}')
        sources['ethiopia_holdout'].append(r)
    for key,path in (
        ('original_train',args.original_train),('original_val',args.original_val),
        ('jic_val',args.jic_val),('iari_val',Path(manifest['iari_original_validation'])),
        ('iari_test',Path(manifest['iari_original_test']))):
        print(f'Hashing {key}...', flush=True)
        if key.startswith('iari'):
            mapping={'Healthy':'control','Brown_rust':'diseased'}
            sources[key] = []
            for label,folder in mapping.items():
                for fp in image_files(path / folder):
                    sources[key].append(make_record(fp,key,label))
        else:
            sources[key]=scan(path,key,CLASSES)
    # Stage-to-protected and stage-to-existing-training flags.
    print('Comparing image hashes (no files will be changed)...', flush=True)
    flags=[]
    for key in ('original_train','original_val','jic_val','iari_val','iari_test','ethiopia_holdout'):
        purpose='training_redundancy' if key=='original_train' else 'training_vs_holdout'
        flags += compare(sources['staged_train'], sources[key], args.phash_threshold, purpose)
    # New evaluation holdouts should not duplicate images used for original-model fitting.
    new_holdouts=sources['iari_val']+sources['iari_test']+sources['ethiopia_holdout']
    for key in ('original_train','original_val','jic_val'):
        flags += compare(new_holdouts, sources[key], args.phash_threshold, 'new_holdout_vs_existing_data')
    cols=('purpose','candidate_source','candidate_class','candidate_path',
          'matched_source','matched_class','matched_path','exact_sha256_match',
          'phash_distance','cross_class')
    # Report exact vs perceptual-only distinctly.
    flagged_candidates={purpose:len({r['candidate_path'] for r in flags if r['purpose']==purpose})
                        for purpose in ('training_redundancy','training_vs_holdout','new_holdout_vs_existing_data')}
    paircounts=Counter((x['purpose'],x['matched_source'], 'exact' if x['exact_sha256_match'] else 'phash_only') for x in flags)
    prior_holdout = manifest.get('training_candidates_excluded_due_to_holdout_similarity', [])
    prior_within = manifest.get('training_candidates_excluded_due_to_training_similarity', [])
    prior_exclusions={
        'staging_holdout_exact':sum(bool(x.get('exact')) for x in prior_holdout),
        'staging_holdout_phash_only':sum(not bool(x.get('exact')) for x in prior_holdout),
        'staging_train_exact':sum(bool(x.get('exact')) for x in prior_within),
        'staging_train_phash_only':sum(not bool(x.get('exact')) for x in prior_within),
    }
    summary={
        'status':'exploratory image-similarity audit; flagged matches require manual review',
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'phash_threshold':args.phash_threshold,
        'image_counts':{k:len(v) for k,v in sources.items()},
        'prior_staging_exclusion_types':prior_exclusions,
        'flagged_candidate_images_by_purpose':flagged_candidates,
        'pair_count_by_purpose_and_target':[
            {'purpose':k[0], 'matched_source':k[1], 'match_type':k[2], 'pair_count':v}
            for k,v in sorted(paircounts.items())],
        'cross_class_flagged_pairs':sum(x['cross_class'] for x in flags),
        'limitations':[
            'pHash flags can be false positives; review both images visually before excluding.',
            'Absence of pHash matches does not rule out same-plant, same-field or shared-session relationships.',
            'The JIC evaluation was already examined previously and is not a blind test.',
            'Do not select hyperparameters or conditions based on protected test performance.',
            'This command does not filter staged data or modify any source images.'
        ]
    }
    output.mkdir(parents=True)
    write_csv(output/'all_similarity_flags.csv',flags,cols)
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('limitations','created_utc')},indent=2))
    print(f'\nSaved read-only audit to {output}')

if __name__=='__main__':
    main()
