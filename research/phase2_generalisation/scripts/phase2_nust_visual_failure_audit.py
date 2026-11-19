"""NUST visual failure audit (read-only source images, model not loaded).

Run from project root:
    python phase2_nust_visual_failure_audit.py

Outputs a PNG contact sheet and manifest CSV under outputs/phase2_nust_evaluation.
Selections are descriptive/illustrative, not a new evaluation sample.
"""
from pathlib import Path
import csv
import math

import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont

from phase2_config import NUST_REFERENCE as CSV, OUT as OUTBASE, nust_path
OUT = OUTBASE / 'phase2_nust_evaluation'
OUT.mkdir(parents=True, exist_ok=True)

with CSV.open(newline='', encoding='utf-8-sig') as fh:
    rows = list(csv.DictReader(fh))
if not rows:
    raise ValueError(f'No predictions in {CSV}')

for row in rows:
    row['confidence_f'] = float(row['confidence'])
    row['coverage_f'] = float(row['lesion_coverage_pct'])

# Predefined descriptive strata; not a statistically representative selection.
# Up to 4 in each group, without duplicate photographs.
groups = [
    ('High-confidence Healthy', sorted((r for r in rows if r['predicted_label']=='Healthy'), key=lambda r: (-r['confidence_f'], r['image_path']))),
    ('Brown-rust predictions', sorted((r for r in rows if r['predicted_label']=='Brown_rust'), key=lambda r: (-r['confidence_f'], r['image_path']))),
    ('Largest lesion coverage', sorted(rows, key=lambda r: (-r['coverage_f'], r['image_path']))),
]
selected=[]
seen=set()
for group, candidates in groups:
    count=0
    for row in candidates:
        if row['image_path'] in seen:
            continue
        selected.append((group, row))
        seen.add(row['image_path'])
        count+=1
        if count==4:
            break

if not selected:
    raise ValueError('No images selected')

FONT=ImageFont.load_default()
# Each row: original overview, lesion overlay, model-size input (upscaled for inspection)
CW,CH=340,235
HEADER=57
ROW_H=CH+HEADER
BG='#f4f5f7'
canvas=Image.new('RGB',(CW*3, ROW_H*len(selected)+54), 'white')
draw=ImageDraw.Draw(canvas)
draw.text((12,9),'NUST failure audit — original / annotated lesions / 224×224 input',fill='black',font=FONT)
draw.text((12,28),'Orange areas: nonzero mask pixels. Examples selected descriptively, not randomly.',fill='#444',font=FONT)

def contain(im, w, h):
    out=Image.new('RGB',(w,h),BG)
    scaled=ImageOps.contain(im,(w-12,h-12),method=Image.Resampling.LANCZOS)
    out.paste(scaled,((w-scaled.width)//2,(h-scaled.height)//2))
    return out

manifest=[]
for idx,(group,r) in enumerate(selected):
    ip=nust_path(r['image_path']); mp=nust_path(r['mask_path'])
    if not ip.is_file() or not mp.is_file():
        raise FileNotFoundError(f'Missing original image or mask: {ip} | {mp}')
    with Image.open(ip) as src:
        orig=ImageOps.exif_transpose(src).convert('RGB')
    with Image.open(mp) as src:
        mask=src.convert('L')
    if mask.size != orig.size:
        raise ValueError(f'Size mismatch for {ip.name}: image {orig.size}, mask {mask.size}. Inspect orientation before continuing.')
    # Prepare model view from the original image, before downsampling for the overlay.
    model_view=orig.resize((224,224),Image.Resampling.BILINEAR)
    # Downsample ONLY for visualization to avoid high-resolution memory pressure.
    thumbnail=ImageOps.contain(orig,(1000,1000),method=Image.Resampling.LANCZOS)
    mask=mask.resize(thumbnail.size, Image.Resampling.NEAREST)
    arr=np.asarray(mask)
    lesion=(arr>127).astype('uint8')*190
    alpha=Image.fromarray(lesion,'L')
    tint=Image.new('RGB',thumbnail.size,(250,104,28))
    over=Image.composite(tint,thumbnail,alpha)
    # Model-size view is upscaled for human inspection. No additional normalization.
    ims=[contain(thumbnail,CW,CH),contain(over,CW,CH),contain(model_view,CW,CH)]
    y=54+idx*ROW_H
    draw.rectangle((0,y,CW*3,y+ROW_H-1),fill='#fafafa' if idx%2==0 else '#efefef')
    info=f'{group} | {r["split"]}/{r["image_name"]} | predicted {r["predicted_label"]} ({r["confidence_f"]:.1%}) | lesion {r["coverage_f"]:.3f}%'
    draw.text((10,y+4),info[:150],fill='black',font=FONT)
    for j,im in enumerate(ims):
        canvas.paste(im,(j*CW,y+HEADER-CH//10))
    manifest.append({k:r[k] for k in ['split','image_name','image_path','mask_path','predicted_label','confidence','prob_brown_rust','prob_healthy','prob_yellow_rust','lesion_coverage_pct']} | {'selection_group':group})

png=OUT/'nust_visual_failure_audit.png'
manifest_path=OUT/'nust_visual_audit_manifest.csv'
canvas.save(png,optimize=True)
with manifest_path.open('w',newline='',encoding='utf-8') as fh:
    w=csv.DictWriter(fh,fieldnames=list(manifest[0].keys()))
    w.writeheader();w.writerows(manifest)
print(f'Rows read: {len(rows)}')
print(f'Examples illustrated: {len(selected)}')
print(f'Saved: {png}')
print(f'Saved: {manifest_path}')
print('NOTE: This is a descriptive visual audit. Do not infer causation or change the evaluation cohort based on these examples.')
