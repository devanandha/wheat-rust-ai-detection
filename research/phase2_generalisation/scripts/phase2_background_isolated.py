"""Wheat Rust AI Phase 2 selected-leaf masking pilot, crash-isolated.

From repository root:
    python -u phase2_background_isolated.py

Runs each of nine images in a separate Python process to isolate TensorFlow
native crashes. NEVER changes model, masks or frozen reference predictions.
Uses original RGB 0..255, TensorFlow bilinear resize to 224x224, and validates
all three scores against frozen JIC predictions before evaluating a mask.
The pilot is outcome-selected and cannot isolate natural background causation.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from phase2_config import JIC_DIR, MODEL as CONFIG_MODEL, JIC_REFERENCE, OUT as OUTBASE

ROOT = JIC_DIR
MODEL = CONFIG_MODEL
REF = JIC_REFERENCE
OUT = OUTBASE / 'phase2_background_pilot'
MASK_DIR = OUT / 'masks'
CLASSES = ('Brown_rust', 'Healthy', 'Yellow_rust')
SELECTED = [
 'Yellow_rust/YellowRust2280.png',
 'Yellow_rust/YellowRust2292.png',
 'Yellow_rust/YellowRust2324.png',
 'Yellow_rust/YellowRust2405.png',
 'Yellow_rust/YellowRust3037.png',
 'Yellow_rust/YellowRust3422.png',
 'Yellow_rust/YellowRust40.png',
 'Yellow_rust/YellowRust626.png',
 'Yellow_rust/YellowRust3856.png',
]

def worker(index: int):
    # Import TF ONLY in worker processes.
    import numpy as np
    import pandas as pd
    from PIL import Image
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    rel = SELECTED[index]
    src = ROOT.joinpath(*rel.split('/'))
    mask_path = MASK_DIR / (Path(rel).stem + '_mask.png')
    for f in (src, mask_path, MODEL, REF):
        if not f.is_file():
            raise FileNotFoundError(f)
    reference = pd.read_csv(REF).set_index('image_id')
    if not reference.index.is_unique:
        raise RuntimeError('Frozen reference contains duplicate image IDs')
    row_ref = reference.loc[rel]
    if row_ref.actual != 'Yellow_rust':
        raise RuntimeError(f'Unexpected true label: {rel}')
    print(f'[{index+1}/9] Loading model for {Path(rel).name}', flush=True)
    model = tf.keras.models.load_model(MODEL, compile=False)
    if model.output_shape[-1] != 3:
        raise ValueError('Model must output three class scores')

    raw = tf.io.read_file(str(src))
    rgb = tf.io.decode_image(raw, channels=3, expand_animations=False)
    rgb.set_shape((None, None, 3))
    x = tf.image.resize(rgb, (224, 224), method='bilinear', antialias=False)
    original_scores = np.asarray(model(x[None, ...], training=False), dtype=np.float64)[0]
    old = np.asarray([float(row_ref['score_' + c]) for c in CLASSES], dtype=np.float64)
    error = float(np.max(np.abs(original_scores - old)))
    print(f'[{index+1}/9] Baseline max difference: {error:.9g}', flush=True)
    if error > 1e-3:
        raise RuntimeError('STOP: baseline does not reproduce frozen JIC class scores')

    with Image.open(src) as im:
        original_size = im.size
    with Image.open(mask_path) as im:
        if im.size != original_size:
            raise RuntimeError('Mask size does not match original image')
        mask = np.asarray(im.convert('L').resize((224, 224), Image.Resampling.NEAREST), dtype=np.uint8) >= 128
    coverage = float(mask.mean())
    if not 0.01 <= coverage <= 0.95:
        raise RuntimeError(f'Unusual foreground fraction: {coverage:.1%}')
    x_np = np.asarray(x, dtype=np.float32)
    masked = np.where(mask[..., None], x_np, np.array([128.,128.,128.], dtype=np.float32)).astype(np.float32)
    modified_scores = np.asarray(model(tf.convert_to_tensor(masked[None, ...]), training=False), dtype=np.float64)[0]
    pred_orig = CLASSES[int(np.argmax(original_scores))]
    pred_new = CLASSES[int(np.argmax(modified_scores))]
    result = dict(image_id=rel, actual=row_ref.actual, foreground_fraction=coverage,
                  baseline_predicted=pred_orig, background_replaced_predicted=pred_new,
                  baseline_correct=int(pred_orig == row_ref.actual),
                  background_replaced_correct=int(pred_new == row_ref.actual),
                  max_baseline_reference_score_difference=error)
    for j, cls in enumerate(CLASSES):
        result['baseline_' + cls] = float(original_scores[j])
        result['background_replaced_' + cls] = float(modified_scores[j])
    part = OUT / 'isolated_results'
    part.mkdir(parents=True, exist_ok=True)
    target = part / (Path(rel).stem + '.json')
    target.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'[{index+1}/9] {pred_orig} -> {pred_new}; saved {target}', flush=True)


def orchestrate():
    import pandas as pd
    part = OUT / 'isolated_results'
    part.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['TF_ENABLE_ONEDNN_OPTS'] = '0'
    env['TF_CPP_MIN_LOG_LEVEL'] = '2'
    failures = []
    script = str(Path(__file__).resolve())
    for i, rel in enumerate(SELECTED):
        target = part / (Path(rel).stem + '.json')
        if target.exists():
            try:
                saved = json.loads(target.read_text(encoding='utf-8'))
                if saved.get('image_id') == rel:
                    print(f'[{i+1}/9] Already saved: {Path(rel).name}', flush=True)
                    continue
            except Exception:
                pass
        print(f'Launching isolated image {i+1}/9: {Path(rel).name}', flush=True)
        outcome = subprocess.run([sys.executable, '-u', script, '--index', str(i)], env=env)
        if outcome.returncode != 0 or not target.is_file():
            failures.append({'image_id':rel,'exit_code':outcome.returncode})
            print(f'FAILED {rel}: return code {outcome.returncode}', flush=True)
    records = []
    for rel in SELECTED:
        path = part / (Path(rel).stem + '.json')
        if path.is_file():
            obj = json.loads(path.read_text(encoding='utf-8'))
            if obj.get('image_id') == rel:
                records.append(obj)
    if records:
        frame = pd.DataFrame(records)
        frame.to_csv(OUT / 'paired_background_pilot_isolated.csv', index=False)
        print(f'Saved {len(records)}/9 results: {OUT / "paired_background_pilot_isolated.csv"}',flush=True)
    result = {
        'research_status':'exploratory_outcome_selected_nine_image_pilot_not_independent',
        'n_completed':len(records), 'n_expected':len(SELECTED), 'failures':failures,
        'note':'Masks isolate selected leaves, not all foliage. Grey replacement creates artificial edges. No claim of background causation.',
    }
    if len(records) == 9:
        frame = pd.DataFrame(records)
        result.update(changed_predictions=int((frame.baseline_predicted != frame.background_replaced_predicted).sum()),
                      wrong_to_right=int(((frame.baseline_correct==0)&(frame.background_replaced_correct==1)).sum()),
                      right_to_wrong=int(((frame.baseline_correct==1)&(frame.background_replaced_correct==0)).sum()))
    (OUT/'pilot_summary_isolated.json').write_text(json.dumps(result, indent=2),encoding='utf-8')
    print(json.dumps(result, indent=2),flush=True)

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--index', type=int, choices=range(9), default=None)
    a=parser.parse_args()
    if a.index is None:
        orchestrate()
    else:
        worker(a.index)
