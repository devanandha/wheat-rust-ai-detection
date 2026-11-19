"""Exploratory paired centre-crop sensitivity experiment, Wheat Rust AI Phase 2.

Run from repository root:
    python phase2_cropping_experiment.py

Uses src.data.load_split with the exact audit preprocessing. The unchanged model
must reproduce ALL stored JIC per-image class scores before the experiment runs.
This is a sensitivity test, NOT evidence that background is the causal mechanism.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from src.data import EXPECTED_CLASSES, IMAGE_SIZE, load_split

CLASSES = ("Brown_rust", "Healthy", "Yellow_rust")
from phase2_config import JIC_DIR, MODEL as CONFIG_MODEL, JIC_REFERENCE, OUT as OUTBASE

ROOT = JIC_DIR
MODEL = CONFIG_MODEL
REFERENCE = JIC_REFERENCE
OUT = OUTBASE / 'phase2_cropping_experiment'
CROP_FRACTIONS = (0.8, 0.6)  # prespecified exploratory transformations
BATCH_SIZE = 2
SCORE_TOLERANCE = 1e-3


def analyse(rows: pd.DataFrame, prefix: str) -> dict:
    output = {}
    for scope, d in (("all", rows), ("yellow", rows[rows.actual == "Yellow_rust"])):
        correct_orig = d.baseline_predicted.eq(d.actual)
        correct_crop = d.crop_predicted.eq(d.actual)
        output[scope] = {
            "n": int(len(d)),
            "baseline_accuracy": float(correct_orig.mean()),
            "crop_accuracy": float(correct_crop.mean()),
            "changed_predictions": int((d.baseline_predicted != d.crop_predicted).sum()),
            "wrong_to_right": int((~correct_orig & correct_crop).sum()),
            "right_to_wrong": int((correct_orig & ~correct_crop).sum()),
            "mean_yellow_score_delta": float((d.crop_Yellow_rust - d.baseline_Yellow_rust).mean()),
            "crop_predicted_class_counts": {c: int((d.crop_predicted == c).sum()) for c in CLASSES},
        }
    return output


def main():
    if tuple(EXPECTED_CLASSES) != CLASSES or tuple(IMAGE_SIZE) != (224, 224):
        raise RuntimeError("Class order or image size differs from saved audit assumptions")
    for path in (ROOT, MODEL, REFERENCE):
        if not path.exists():
            raise FileNotFoundError(f"Required input missing: {path}")
    ref = pd.read_csv(REFERENCE)
    if len(ref) != 489 or ref.image_id.duplicated().any():
        raise RuntimeError("Unexpected reference CSV size or duplicate image ids")
    ref_by_id = ref.set_index("image_id")
    model = tf.keras.models.load_model(MODEL, compile=False)
    ds, filepaths = load_split(ROOT, shuffle=False, batch_size=BATCH_SIZE, return_paths=True)
    result_rows = []
    seen = 0
    for tensors, onehot in ds:
        x = tf.cast(tensors, tf.float32)
        y = np.argmax(np.asarray(onehot), axis=1)
        n = len(y)
        base = np.asarray(model(x, training=False), dtype=float)
        crops = {}
        for frac in CROP_FRACTIONS:
            # Crop the already-loaded image tensor. Original RGB 0..255 range is unchanged.
            cropped = tf.image.central_crop(x, central_fraction=frac)
            cropped = tf.image.resize(cropped, IMAGE_SIZE, method="bilinear")
            crops[frac] = np.asarray(model(cropped, training=False), dtype=float)
        for i in range(n):
            image_id = Path(filepaths[seen+i]).relative_to(ROOT).as_posix()
            if image_id not in ref_by_id.index:
                raise RuntimeError(f"Image not in frozen predictions: {image_id}")
            r = ref_by_id.loc[image_id]
            if CLASSES[y[i]] != r.actual:
                raise RuntimeError(f"Label mismatch: {image_id}")
            reference_scores = np.array([r['score_'+c] for c in CLASSES], dtype=float)
            if not np.allclose(base[i], reference_scores, rtol=0, atol=SCORE_TOLERANCE):
                raise RuntimeError(
                    f"STOP: baseline score mismatch for {image_id}. "
                    f"Max difference={np.max(abs(base[i]-reference_scores)):.6g}; "
                    "check preprocessing, model and dataset versions before interpreting crops."
                )
            for frac in CROP_FRACTIONS:
                p = crops[frac][i]
                row = {
                    'image_id': image_id,
                    'actual': r.actual,
                    'crop_fraction': frac,
                    'baseline_predicted': CLASSES[int(base[i].argmax())],
                    'crop_predicted': CLASSES[int(p.argmax())],
                    'baseline_max_score': float(base[i].max()),
                    'crop_max_score': float(p.max()),
                }
                for j, c in enumerate(CLASSES):
                    row['baseline_'+c] = float(base[i,j])
                    row['crop_'+c] = float(p[j])
                result_rows.append(row)
        seen += n
        print(f"Validated and evaluated {seen}/{len(filepaths)} images", end='\r', flush=True)
    if seen != len(filepaths) or seen != len(ref):
        raise RuntimeError(f"Input count mismatch: {seen} versus {len(ref)}")
    print(f"\nBaseline verified against stored probabilities for {seen} images.")
    OUT.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(result_rows)
    results.to_csv(OUT/'paired_crop_predictions.csv', index=False)
    summary = {
        'research_status':'exploratory_reused_JIC_set',
        'baseline_validation':'all 489 per-image class scores matched audit within absolute tolerance 0.001',
        'model':str(MODEL), 'data':str(ROOT), 'crop_fractions':list(CROP_FRACTIONS),
        'caveats':[
            'The JIC dataset has been repeatedly inspected; this is NOT an independent holdout.',
            'Centre cropping changes leaf scale, context and framing simultaneously; background causation is NOT established.',
            'A crop can remove disease symptoms; changes in accuracy are not automatically improvements in robustness.',
            'No model weights or calibrations were changed.',
        ],
        'conditions':{},
    }
    for frac in CROP_FRACTIONS:
        s = results[results.crop_fraction.eq(frac)]
        summary['conditions'][str(frac)] = analyse(s, str(frac))
        print(f"\nCROP {frac}: {json.dumps(summary['conditions'][str(frac)], indent=2)}")
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2),encoding='utf-8')
    print(f"\nSaved: {OUT/'paired_crop_predictions.csv'}")
    print(f"Saved: {OUT/'summary.json'}")

if __name__ == '__main__':
    main()
