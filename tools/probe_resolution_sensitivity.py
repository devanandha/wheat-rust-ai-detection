"""Exploratory paired inference: baseline versus downsample/upsample input.

This is a sensitivity diagnostic, NOT an estimate of real-world generalisation
and NOT evidence that image-resolution cues caused the historic failures.
The model is loaded read-only, and no original image is modified.
Run from the repository root with: python -m tools.probe_resolution_sensitivity
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

CLASSES = ("Brown_rust", "Healthy", "Yellow_rust")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paired_summary(actual: np.ndarray, baseline: np.ndarray,
                   degraded: np.ndarray) -> dict:
    """Compute transparent paired metrics; all inputs are class indices 0..2."""
    actual = np.asarray(actual, dtype=np.int64)
    baseline = np.asarray(baseline, dtype=np.int64)
    degraded = np.asarray(degraded, dtype=np.int64)
    if actual.ndim != 1 or actual.size == 0 or baseline.shape != actual.shape or degraded.shape != actual.shape:
        raise ValueError("Expected three same-length nonempty 1-D arrays")
    if any(np.any((v < 0) | (v >= len(CLASSES))) for v in (actual, baseline, degraded)):
        raise ValueError("Class indices must be between 0 and 2")

    before_correct = baseline == actual
    after_correct = degraded == actual

    def results(predicted: np.ndarray) -> dict:
        matrix = np.zeros((3, 3), dtype=int)
        for truth, pred in zip(actual, predicted):
            matrix[int(truth), int(pred)] += 1
        support = matrix.sum(axis=1)
        recalls = {
            cls: (float(matrix[i, i] / support[i]) if support[i] else None)
            for i, cls in enumerate(CLASSES)
        }
        return {
            "correct": int(np.count_nonzero(predicted == actual)),
            "accuracy": float(np.mean(predicted == actual)),
            "support_by_class": {c: int(support[i]) for i, c in enumerate(CLASSES)},
            "recall_by_class": recalls,
            "confusion_matrix_true_rows_predicted_columns": matrix.tolist(),
        }

    return {
        "count": int(actual.size),
        "baseline": results(baseline),
        "perturbed": results(degraded),
        "accuracy_change_percentage_points": round(
            100.0 * float(np.mean(after_correct) - np.mean(before_correct)), 5
        ),
        "predictions_changed": int(np.count_nonzero(baseline != degraded)),
        "correct_to_wrong": int(np.count_nonzero(before_correct & ~after_correct)),
        "wrong_to_correct": int(np.count_nonzero(~before_correct & after_correct)),
    }


def analyse_source(name: str, root: Path, model, tf, output_dir: Path, batch_size: int) -> dict:
    # Imports here allow unit tests of the reporting calculations without TensorFlow.
    from src.data import EXPECTED_CLASSES, IMAGE_SIZE, load_split

    if tuple(EXPECTED_CLASSES) != CLASSES or tuple(IMAGE_SIZE) != (224, 224):
        raise RuntimeError("Class order or image-size contract differs from saved analysis")
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {root}")

    ds, paths = load_split(root, shuffle=False, batch_size=batch_size, return_paths=True)
    current = 0
    records: list[dict] = []
    y_true: list[int] = []
    pred_baseline: list[int] = []
    pred_perturbed: list[int] = []

    for pixels, labels in ds:
        # src.data.load_split yields resized RGB pixels on [0,255].
        # The saved model contains MobileNetV2 preprocessing internally.
        # Do NOT divide by 255 or call preprocess_input again.
        baseline = np.asarray(model(pixels, training=False))
        # Controlled artificial low-pass perturbation. Neither the model nor
        # source images nor the image labels are changed.
        small = tf.image.resize(pixels, (112, 112), method="bilinear", antialias=False)
        restored = tf.image.resize(small, (224, 224), method="bilinear", antialias=False)
        shifted = np.asarray(model(restored, training=False))
        truth = np.argmax(np.asarray(labels), axis=1)
        if baseline.shape != shifted.shape or baseline.shape != (len(truth), len(CLASSES)):
            raise RuntimeError(f"Unexpected prediction output shape: {baseline.shape}")
        pb = np.argmax(baseline, axis=1)
        ps = np.argmax(shifted, axis=1)

        for i in range(len(truth)):
            filepath = Path(paths[current + i])
            try:
                relative = filepath.relative_to(root)
                image_id = relative.as_posix()
            except ValueError:
                image_id = f"{CLASSES[int(truth[i])]}/{filepath.name}"
            cls = int(truth[i])
            records.append({
                "dataset": name, "image_id": image_id, "actual_class": CLASSES[cls],
                "baseline_prediction": CLASSES[int(pb[i])],
                "perturbed_prediction": CLASSES[int(ps[i])],
                "baseline_correct": int(pb[i] == cls),
                "perturbed_correct": int(ps[i] == cls),
                "baseline_true_class_probability": round(float(baseline[i, cls]), 8),
                "perturbed_true_class_probability": round(float(shifted[i, cls]), 8),
            })
        current += len(truth)
        y_true.extend(map(int, truth))
        pred_baseline.extend(map(int, pb))
        pred_perturbed.extend(map(int, ps))

    if current != len(paths):
        raise RuntimeError(f"Image order/count mismatch: {current} processed versus {len(paths)} file paths")
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset", "image_id", "actual_class", "baseline_prediction",
        "perturbed_prediction", "baseline_correct", "perturbed_correct",
        "baseline_true_class_probability", "perturbed_true_class_probability",
    ]
    with (output_dir / f"{name}_paired_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    summary = paired_summary(np.array(y_true), np.array(pred_baseline), np.array(pred_perturbed))
    summary["mean_change_in_true_class_probability"] = round(
        float(np.mean([
            row["perturbed_true_class_probability"] - row["baseline_true_class_probability"]
            for row in records
        ])), 8
    )
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Previously trained .keras model")
    parser.add_argument("--original-val", required=True, type=Path,
                        help="Original validation class-folder root, not its parent")
    parser.add_argument("--jic-val", required=True, type=Path,
                        help="Previously inspected JIC 3-class evaluation root")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("batch-size must be positive")
    if not args.model.is_file():
        raise SystemExit(f"Model file not found: {args.model}")
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit("TensorFlow is not installed. Install a compatible version; see requirements.txt.") from exc

    model = tf.keras.models.load_model(args.model, compile=False)
    if len(model.output_shape) != 2 or model.output_shape[-1] != 3:
        raise SystemExit(f"Expected 3-class classifier, got {model.output_shape}")

    result = {
        "status": "exploratory_paired_sensitivity_no_retraining",
        "model_sha256": file_sha256(args.model),
        "tensorflow_version": tf.__version__,
        "class_order": list(CLASSES),
        "perturbation": "224x224 RGB resized bilinear to 112x112, bilinear back to 224x224",
        "warnings": [
            "Artificial degradation does not recreate the original source image size or acquisition conditions.",
            "The saved model has built-in preprocessing: no external scaling was applied.",
            "The original validation and JIC sets have been examined repeatedly: results are exploratory, not a new blind test.",
            "Changes in confidence/accuracy do not establish the model uses a resolution shortcut.",
            "No model was trained or changed; original image files were not modified.",
        ],
        "sources": {},
    }
    for name, root in [("original_val", args.original_val), ("jic", args.jic_val)]:
        print(f"Evaluating {name} with saved model...", flush=True)
        result["sources"][name] = analyse_source(
            name, root, model, tf, args.output_dir, args.batch_size
        )

    result["historical_baseline_check"] = {
        "original_val_correct_expected": 728,
        "jic_correct_expected": 188,
        "original_val_matches": result["sources"]["original_val"]["baseline"]["correct"] == 728,
        "jic_matches": result["sources"]["jic"]["baseline"]["correct"] == 188,
    }
    if not all([result["historical_baseline_check"]["original_val_matches"],
                result["historical_baseline_check"]["jic_matches"]]):
        result["warnings"].append(
            "Baseline did not reproduce historical figures. Investigate saved model, class folders, "
            "preprocessing and TensorFlow version BEFORE interpreting perturbation effects."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "paired_sensitivity_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "baseline_correct": {name: data["baseline"]["correct"] for name, data in result["sources"].items()},
        "baseline_matches_history": {
            name: result["historical_baseline_check"][name + "_matches"]
            for name in ("original_val", "jic")
        },
        "prediction_changes": {name: data["predictions_changed"] for name, data in result["sources"].items()},
        "output": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
