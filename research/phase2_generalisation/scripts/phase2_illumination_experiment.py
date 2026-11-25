"""Wheat Rust AI Phase 2: exploratory illumination sensitivity on the JIC validation set.

Run from the repository root:
    python -u phase2_illumination_experiment.py

Uses the EXISTING src.data.load_split to reproduce the saved baseline. Brightness
changes are applied to the already-loaded 224x224 RGB [0,255] tensors, before the
saved model's internal preprocessing. No weights are changed.

JIC has already been inspected during development; results are EXPLORATORY and
cannot establish independent field generalisation or illuminate the causal
origin of any domain shift. Multiplication and clipping change pixel statistics.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# Set these BEFORE importing TensorFlow; helps some Windows CPU installations.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from src.data import EXPECTED_CLASSES, IMAGE_SIZE, load_split

CLASSES = ("Brown_rust", "Healthy", "Yellow_rust")
from phase2_config import JIC_DIR, MODEL as CONFIG_MODEL, JIC_REFERENCE, OUT as OUTBASE

ROOT = JIC_DIR
MODEL = CONFIG_MODEL
REFERENCE = JIC_REFERENCE
OUT = OUTBASE / 'phase2_illumination_experiment'
PARTIAL = OUT / "paired_illumination_predictions.partial.csv"
FINAL = OUT / "paired_illumination_predictions.csv"
SUMMARY = OUT / "summary.json"
BATCH_SIZE = 2
# Fixed before experiment; NOT tuned on the JIC labels.
BRIGHTNESS_FACTORS = (0.8, 1.2)
SCORE_TOLERANCE = 1e-3

FIELDS = ["image_id", "actual", "brightness_factor", "baseline_predicted",
          "modified_predicted", "baseline_max_score", "modified_max_score"]
for prefix in ("baseline", "modified"):
    FIELDS.extend(prefix + "_" + c for c in CLASSES)


def summarize(d: pd.DataFrame) -> dict:
    result = {}
    for name, subset in (("all", d),
                         ("yellow", d[d.actual.eq("Yellow_rust")])):
        before = subset.baseline_predicted.eq(subset.actual)
        after = subset.modified_predicted.eq(subset.actual)
        false_healthy_before = ((subset.actual != "Healthy") &
                                (subset.baseline_predicted == "Healthy"))
        false_healthy_after = ((subset.actual != "Healthy") &
                               (subset.modified_predicted == "Healthy"))
        result[name] = {
            "n": int(len(subset)),
            "baseline_accuracy": float(before.mean()),
            "modified_accuracy": float(after.mean()),
            "changed_predictions": int((subset.baseline_predicted != subset.modified_predicted).sum()),
            "wrong_to_right": int((~before & after).sum()),
            "right_to_wrong": int((before & ~after).sum()),
            "false_healthy_before": int(false_healthy_before.sum()),
            "false_healthy_after": int(false_healthy_after.sum()),
            "high_confidence_errors_before": int((~before & (subset.baseline_max_score >= 0.9)).sum()),
            "high_confidence_errors_after": int((~after & (subset.modified_max_score >= 0.9)).sum()),
            "mean_yellow_score_change": float(
                (subset.modified_Yellow_rust - subset.baseline_Yellow_rust).mean()),
            "modified_prediction_counts": {
                c: int((subset.modified_predicted == c).sum()) for c in CLASSES},
        }
    return result


def main() -> None:
    if tuple(EXPECTED_CLASSES) != CLASSES or tuple(IMAGE_SIZE) != (224, 224):
        raise RuntimeError("STOP: Class order or image size differs from audit assumptions.")
    for path in (ROOT, MODEL, REFERENCE):
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")
    ref = pd.read_csv(REFERENCE)
    if len(ref) != 489 or ref.image_id.duplicated().any():
        raise RuntimeError("STOP: unexpected reference size or duplicate image IDs.")
    lookup = ref.set_index("image_id")
    OUT.mkdir(parents=True, exist_ok=True)

    completed = set()
    if PARTIAL.exists():
        previous = pd.read_csv(PARTIAL)
        if list(previous.columns) != FIELDS:
            raise RuntimeError("Partial results have unexpected columns. Move them aside before rerunning.")
        completed = set(zip(previous.image_id, previous.brightness_factor.round(4)))
        if previous.duplicated(["image_id", "brightness_factor"]).any():
            raise RuntimeError("Duplicate rows in partial results; investigate before resuming.")
        print(f"Resuming; {len(completed)} image/brightness pairs already saved.", flush=True)

    # Limit TensorFlow thread counts to reduce native CPU resource contention.
    try:
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:
        pass

    print("Loading frozen classifier...", flush=True)
    model = tf.keras.models.load_model(MODEL, compile=False)
    ds, paths = load_split(ROOT, shuffle=False, batch_size=BATCH_SIZE, return_paths=True)
    if len(paths) != 489:
        raise RuntimeError(f"STOP: expected 489 JIC files, found {len(paths)}")

    with PARTIAL.open("a", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        if output.tell() == 0:
            writer.writeheader()
            output.flush()
        offset = 0
        for images, labels in ds:
            x = tf.cast(images, tf.float32)
            n = int(x.shape[0])
            y = np.argmax(np.asarray(labels), axis=1)
            # Even when resuming, ensure the baseline for each image still matches.
            base = np.asarray(model(x, training=False), dtype=np.float64)
            modified = {}
            needed = []
            for j in range(n):
                image_id = Path(paths[offset + j]).relative_to(ROOT).as_posix()
                if image_id not in lookup.index:
                    raise RuntimeError(f"STOP: missing reference image {image_id}")
                r = lookup.loc[image_id]
                if CLASSES[int(y[j])] != r.actual:
                    raise RuntimeError(f"STOP: label mismatch for {image_id}")
                old = np.array([float(r["score_" + c]) for c in CLASSES])
                if not np.allclose(base[j], old, atol=SCORE_TOLERANCE, rtol=0):
                    raise RuntimeError(f"STOP: baseline mismatch for {image_id}; max difference "
                                       f"{np.max(np.abs(base[j] - old)):.8g}")
                needed.append(image_id)

            for factor in BRIGHTNESS_FACTORS:
                if all((image_id, factor) in completed for image_id in needed):
                    continue
                # Brightness intensity: multiply RGB values and clip to valid range.
                # This also affects contrast/saturation in bright regions.
                transformed = tf.clip_by_value(x * factor, 0.0, 255.0)
                modified[factor] = np.asarray(model(transformed, training=False), dtype=np.float64)

            for j, image_id in enumerate(needed):
                r = lookup.loc[image_id]
                for factor in BRIGHTNESS_FACTORS:
                    if (image_id, factor) in completed:
                        continue
                    p = modified[factor][j]
                    row = {
                        "image_id": image_id,
                        "actual": r.actual,
                        "brightness_factor": factor,
                        "baseline_predicted": CLASSES[int(base[j].argmax())],
                        "modified_predicted": CLASSES[int(p.argmax())],
                        "baseline_max_score": float(base[j].max()),
                        "modified_max_score": float(p.max()),
                    }
                    for k, c in enumerate(CLASSES):
                        row["baseline_" + c] = float(base[j, k])
                        row["modified_" + c] = float(p[k])
                    writer.writerow(row)
                    completed.add((image_id, factor))
            output.flush()  # preserve each completed batch if Windows exits abruptly
            offset += n
            if offset % 20 < n or offset == len(paths):
                print(f"Baseline verified; evaluated {offset}/{len(paths)} images", flush=True)

    if offset != 489 or len(completed) != 489 * len(BRIGHTNESS_FACTORS):
        raise RuntimeError(f"Incomplete: processed {offset} images; saved {len(completed)} pairs")
    results = pd.read_csv(PARTIAL)
    if results.duplicated(["image_id", "brightness_factor"]).any():
        raise RuntimeError("Duplicate output rows detected")
    results = results.sort_values(["image_id", "brightness_factor"]).reset_index(drop=True)
    results.to_csv(FINAL, index=False)
    report = {
        "research_status": "exploratory_reused_JIC_not_independent_holdout",
        "baseline_validation": "All 489 predictions matched stored class scores within 0.001",
        "model": str(MODEL), "image_root": str(ROOT),
        "conditions": {},
        "caveats": [
            "JIC has been repeatedly inspected and is not an untouched test set.",
            "Brightness multiplication changes image intensity and may clip bright pixels.",
            "Intervention sensitivity does not prove illumination caused source shift.",
            "No model training, fine-tuning, or calibration was performed.",
        ],
    }
    for factor in BRIGHTNESS_FACTORS:
        subset = results[np.isclose(results.brightness_factor, factor)]
        report["conditions"][str(factor)] = summarize(subset)
        print(f"\nBrightness factor {factor}:\n"
              f"{json.dumps(report['conditions'][str(factor)], indent=2)}", flush=True)
    SUMMARY.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved: {FINAL}\nSaved: {SUMMARY}", flush=True)


if __name__ == "__main__":
    main()
