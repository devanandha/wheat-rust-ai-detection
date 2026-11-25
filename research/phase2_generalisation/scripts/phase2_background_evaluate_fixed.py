"""Wheat Rust AI Phase 2: nine-image selected-leaf masking PILOT.

Run from repository root:
    python -u phase2_background_evaluate_fixed.py

Does NOT load the complete JIC dataset. Never overwrites original audit or masks.
Baseline scores must match the archived JIC predictions (absolute tolerance 1e-3).
If baseline validation fails, do NOT interpret modified-image scores.
"""
import json
import os
from pathlib import Path

# Optional lower-thread CPU configuration, before importing TensorFlow.
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

from phase2_config import JIC_DIR, MODEL as CONFIG_MODEL, JIC_REFERENCE, OUT as OUTBASE

ROOT = JIC_DIR
MODEL = CONFIG_MODEL
REF = JIC_REFERENCE
MASK_DIR = OUTBASE / 'phase2_background_pilot' / 'masks'
OUT = OUTBASE / 'phase2_background_pilot'
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
FILL_RGB = np.asarray((128, 128, 128), dtype=np.float32)
SCORE_TOLERANCE = 1e-3


def load_original_224(path: Path):
    """TensorFlow RGB decoding + bilinear resizing; RGB remains in 0..255.

    This matches typical image_dataset_from_directory decoding. An archived-score
    comparison below is the *required* check for pipeline equivalence.
    """
    raw = tf.io.read_file(str(path))
    rgb = tf.io.decode_image(raw, channels=3, expand_animations=False)
    rgb.set_shape((None, None, 3))
    return tf.image.resize(rgb, (224, 224), method='bilinear', antialias=False)


def main():
    for f in (MODEL, REF):
        if not f.is_file():
            raise FileNotFoundError(f)
    if not ROOT.is_dir():
        raise FileNotFoundError(ROOT)
    for relative in SELECTED:
        if not ROOT.joinpath(*relative.split('/')).is_file():
            raise FileNotFoundError(ROOT.joinpath(*relative.split('/')))
        mask_path = MASK_DIR / (Path(relative).stem + '_mask.png')
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)

    reference = pd.read_csv(REF).set_index('image_id', verify_integrity=True)
    model = tf.keras.models.load_model(MODEL, compile=False)
    if model.output_shape[-1] != 3:
        raise ValueError('Expected a three-class classifier')
    OUT.mkdir(parents=True, exist_ok=True)
    output_csv = OUT / 'paired_background_pilot_fixed.csv'
    result_json = OUT / 'pilot_summary_fixed.json'
    rows = []
    for index, relative in enumerate(SELECTED, start=1):
        image_file = ROOT.joinpath(*relative.split('/'))
        if relative not in reference.index:
            raise ValueError(f'Missing stored prediction: {relative}')
        r = reference.loc[relative]
        if r.actual != 'Yellow_rust':
            raise ValueError(f'Unexpected class for {relative}: {r.actual}')

        x = load_original_224(image_file)
        base = np.asarray(model(x[None, ...], training=False), dtype=np.float64)[0]
        historic = np.asarray([float(r['score_' + cls]) for cls in CLASSES])
        err = float(np.max(np.abs(base - historic)))
        print(f'[{index}/9] {Path(relative).name}: baseline max score difference {err:.6g}', flush=True)
        if err > SCORE_TOLERANCE:
            raise RuntimeError(
                f'STOP: baseline mismatch on {relative}: {err:.6g} > {SCORE_TOLERANCE}. '
                'Confirm image decoding and resize logic against src.data.load_split; '
                'no masking conclusion is valid until this matches.'
            )

        mask_file = MASK_DIR / (Path(relative).stem + '_mask.png')
        with Image.open(image_file) as im:
            original_size = im.size
        with Image.open(mask_file) as m:
            if m.size != original_size:
                raise ValueError(f'Mask/image dimensions differ: {relative}')
            mask = m.convert('L').resize((224, 224), Image.Resampling.NEAREST)
            foreground = np.asarray(mask, dtype=np.uint8) >= 128
        fraction = float(foreground.mean())
        if not 0.01 <= fraction <= 0.95:
            raise ValueError(f'Unusual coverage ({fraction:.1%}) for {relative}')

        # Mask in same 224x224 space as the verified original image: position and
        # scale of selected foreground pixels remain fixed.
        original_np = np.asarray(x, dtype=np.float32)
        masked_np = np.where(foreground[..., None], original_np, FILL_RGB).astype(np.float32)
        changed = np.asarray(model(tf.convert_to_tensor(masked_np[None, ...]), training=False), dtype=np.float64)[0]
        base_class = CLASSES[int(base.argmax())]
        masked_class = CLASSES[int(changed.argmax())]
        row = {
            'image_id': relative,
            'actual': r.actual,
            'foreground_fraction': fraction,
            'baseline_predicted': base_class,
            'background_replaced_predicted': masked_class,
            'baseline_correct': int(base_class == r.actual),
            'background_replaced_correct': int(masked_class == r.actual),
            'max_baseline_reference_score_difference': err,
        }
        for j, cls in enumerate(CLASSES):
            row['baseline_' + cls] = float(base[j])
            row['background_replaced_' + cls] = float(changed[j])
        rows.append(row)
        # Save after every finished image, so native crash does not erase progress.
        pd.DataFrame(rows).to_csv(output_csv, index=False)
        print(f'[{index}/9] {base_class} -> {masked_class}; partial CSV saved', flush=True)

    frame = pd.DataFrame(rows)
    result = {
        'research_status': 'exploratory_selected_nine_image_pilot_not_independent',
        'n': len(frame),
        'model_unchanged': True,
        'background_fill_rgb': FILL_RGB.astype(int).tolist(),
        'baseline_max_score_difference': max(float(x['max_baseline_reference_score_difference']) for x in rows),
        'changed_predictions': int((frame.baseline_predicted != frame.background_replaced_predicted).sum()),
        'wrong_to_right': int(((frame.baseline_correct == 0) & (frame.background_replaced_correct == 1)).sum()),
        'right_to_wrong': int(((frame.baseline_correct == 1) & (frame.background_replaced_correct == 0)).sum()),
        'warning': 'Outcome-selected nine images; manually drawn masks exclude some other leaves. Grey fill creates unnatural image artifacts. This tests selected-foreground masking sensitivity, NOT isolated background causation or independent performance.',
    }
    result_json.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
    print('Saved:', output_csv, result_json, sep='\n', flush=True)


if __name__ == '__main__':
    main()
