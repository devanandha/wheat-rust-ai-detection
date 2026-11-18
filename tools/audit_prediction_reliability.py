"""Exploratory reliability audit of frozen Wheat Rust AI predictions.

Measures confidence calibration and the danger of false Healthy reassurance.
Historical original-validation and JIC datasets were already inspected; these
outputs CANNOT be claimed as new independent field performance.

Run from repository root: python -m tools.audit_prediction_reliability ...
No model training, file modifications, threshold fitting or external uploads.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

CLASSES = ("Brown_rust", "Healthy", "Yellow_rust")
HEALTHY_INDEX = 1
# A prespecified descriptive grid, NOT fitted or optimised using JIC.
THRESHOLDS = (0.0, 0.5, 0.7, 0.9, 0.95, 0.99)
CALIBRATION_BINS = 10


def model_sha256(file: Path) -> str:
    digest = hashlib.sha256()
    with file.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def classification_metrics(true, prediction) -> dict:
    true = np.asarray(true, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    confusion = np.zeros((3, 3), dtype=np.int64)
    for actual, guessed in zip(true, prediction):
        confusion[int(actual), int(guessed)] += 1
    per_class = {}
    for index, name in enumerate(CLASSES):
        tp = int(confusion[index, index])
        support = int(confusion[index, :].sum())
        predicted = int(confusion[:, index].sum())
        precision = _ratio(tp, predicted)
        recall = _ratio(tp, support)
        f1 = _ratio(2 * tp, support + predicted)
        per_class[name] = {
            "support": support, "precision": precision, "recall": recall, "f1": f1,
        }
    macro_f1 = [v["f1"] for v in per_class.values() if v["support"]]
    recalls = [v["recall"] for v in per_class.values() if v["support"]]
    return {
        "total": int(len(true)),
        "correct": int(np.count_nonzero(true == prediction)),
        "accuracy": _ratio(int(np.count_nonzero(true == prediction)), len(true)),
        "balanced_accuracy": float(np.mean(recalls)) if recalls else None,
        "macro_f1": float(np.mean(macro_f1)) if macro_f1 else None,
        "per_class": per_class,
        "confusion_matrix_rows_true_columns_predicted": confusion.tolist(),
        "diseased_predicted_healthy": int(
            np.count_nonzero((true != HEALTHY_INDEX) & (prediction == HEALTHY_INDEX))
        ),
    }


def reliability_metrics(true, probabilities) -> dict:
    """Pure metrics; no imports of TF/sklearn and no calibrated-confidence claims."""
    y = np.asarray(true, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.ndim != 1 or y.size == 0 or p.shape != (y.size, 3):
        raise ValueError("Need nonempty 1-D labels and Nx3 probabilities.")
    if np.any((y < 0) | (y >= 3)) or not np.all(np.isfinite(p)):
        raise ValueError("Invalid labels or non-finite scores.")
    if np.any(p < 0) or np.any(p > 1) or not np.allclose(p.sum(axis=1), 1, atol=1e-3):
        raise ValueError("Prediction scores must be valid probabilities (approximately summing to 1).")

    pred = p.argmax(axis=1)
    confidence = p.max(axis=1)
    correct = pred == y
    conf_mat = classification_metrics(y, pred)
    one_hot = np.eye(3)[y]
    brier = float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))
    nll = float(np.mean(-np.log(np.clip(p[np.arange(y.size), y], 1e-12, 1.0))))
    bin_rows = []
    ece = 0.0
    for i in range(CALIBRATION_BINS):
        low, high = i / CALIBRATION_BINS, (i + 1) / CALIBRATION_BINS
        mask = (confidence >= low) & (
            confidence < high if i < CALIBRATION_BINS - 1 else confidence <= high
        )
        count = int(mask.sum())
        mean_score = float(confidence[mask].mean()) if count else None
        accuracy = float(correct[mask].mean()) if count else None
        if count:
            ece += count / y.size * abs(accuracy - mean_score)
        bin_rows.append({
            "lower_inclusive": low, "upper": high, "count": count,
            "mean_confidence": mean_score, "observed_accuracy": accuracy,
        })
    selection = []
    for threshold in THRESHOLDS:
        accept = confidence >= threshold
        accepted = int(accept.sum())
        mistakes = int(np.count_nonzero(accept & ~correct))
        false_healthy = int(np.count_nonzero(
            accept & (y != HEALTHY_INDEX) & (pred == HEALTHY_INDEX)
        ))
        accepted_yellow = int(np.count_nonzero(accept & (y == 2)))
        yellow_correct = int(np.count_nonzero(accept & (y == 2) & (pred == 2)))
        selection.append({
            "threshold": threshold,
            "accepted": accepted,
            "referred_for_review": int(y.size - accepted),
            "coverage": float(accepted / y.size),
            "accepted_correct": int(accepted - mistakes),
            "accepted_errors": mistakes,
            "selective_error_rate": _ratio(mistakes, accepted),
            "accepted_diseased_predicted_healthy": false_healthy,
            "accepted_yellow_rust": accepted_yellow,
            "accepted_yellow_rust_correct": yellow_correct,
            "accepted_yellow_rust_recall": _ratio(yellow_correct, accepted_yellow),
        })
    return {
        "classification": conf_mat,
        "mean_max_score": float(confidence.mean()),
        "max_score_ece_10_bins": float(ece),
        "brier_multiclass": brier,
        "negative_log_likelihood": nll,
        "calibration_bins": bin_rows,
        "fixed_threshold_selection": selection,
        "mistakes_with_score_at_least_0_9": int(np.count_nonzero(~correct & (confidence >= 0.9))),
        "mistakes_with_score_at_least_0_99": int(np.count_nonzero(~correct & (confidence >= 0.99))),
    }


def evaluate(name, root: Path, model, output: Path, batch_size: int) -> dict:
    from src.data import EXPECTED_CLASSES, IMAGE_SIZE, load_split
    if tuple(EXPECTED_CLASSES) != CLASSES or tuple(IMAGE_SIZE) != (224, 224):
        raise RuntimeError("Class order/image size differ from audit assumptions.")
    ds, filepaths = load_split(root, shuffle=False, batch_size=batch_size, return_paths=True)
    rows = []
    labels = []
    scores = []
    seen = 0
    for inputs, encoded_labels in ds:
        # RGB 0..255: saved MobileNetV2 model preprocesses internally.
        output_scores = np.asarray(model(inputs, training=False), dtype=np.float64)
        y = np.argmax(np.asarray(encoded_labels), axis=1).astype(int)
        if output_scores.shape != (len(y), len(CLASSES)):
            raise RuntimeError(f"Unexpected output shape: {output_scores.shape}")
        for i, true_index in enumerate(y):
            predicted = int(output_scores[i].argmax())
            image_file = Path(filepaths[seen + i])
            try:
                image_id = image_file.relative_to(root).as_posix()
            except ValueError:
                image_id = f"{CLASSES[true_index]}/{image_file.name}"
            record = {
                "dataset": name, "image_id": image_id,
                "actual": CLASSES[true_index], "predicted": CLASSES[predicted],
                "correct": int(true_index == predicted),
                "max_score": float(output_scores[i].max()),
            }
            for j, cls in enumerate(CLASSES):
                record["score_" + cls] = float(output_scores[i, j])
            rows.append(record)
        labels.extend(y.tolist())
        scores.extend(output_scores.tolist())
        seen += len(y)
    if seen != len(filepaths):
        raise RuntimeError(f"Mismatch: {seen} loaded versus {len(filepaths)} filepaths")

    output.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "image_id", "actual", "predicted", "correct",
              "max_score"] + ["score_" + c for c in CLASSES]
    with (output / f"{name}_predictions.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    result = reliability_metrics(np.asarray(labels), np.asarray(scores))
    with (output / f"{name}_confidence_bins.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(result["calibration_bins"][0]))
        writer.writeheader()
        writer.writerows(result["calibration_bins"])
    with (output / f"{name}_review_tradeoff.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(result["fixed_threshold_selection"][0]))
        writer.writeheader()
        writer.writerows(result["fixed_threshold_selection"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--original-val", type=Path, required=True)
    parser.add_argument("--jic-val", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/reliability_audit"))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    if not args.model.is_file():
        parser.error(f"Missing model: {args.model}")
    for root in (args.original_val, args.jic_val):
        if not root.is_dir():
            parser.error(f"Dataset root missing: {root}")

    # Import TensorFlow only when performing inference, not pure-metric tests.
    import tensorflow as tf
    model = tf.keras.models.load_model(args.model, compile=False)
    if model.output_shape[-1] != 3:
        raise ValueError("Saved model is not a 3-class classifier.")
    result = {
        "status": "exploratory_reused_validation_sets_no_field_test",
        "version": 1,
        "model_sha256": model_sha256(args.model),
        "tensorflow_version": tf.__version__,
        "class_order": list(CLASSES),
        "review_thresholds_fixed_before_new_results": list(THRESHOLDS),
        "warnings": [
            "Reused original validation and JIC data: exploratory, not independent field test.",
            "Confidence and ECE can be misleading with dataset shift and class balance.",
            "Score thresholds are descriptive only; none is an approved autonomous-decision threshold.",
            "False reassurance means a diseased label assigned Healthy; lookalikes and unsupported diseases are not evaluated.",
            "No retraining, score calibration fitting, or data/model modification was performed.",
        ],
        "sources": {},
    }
    for name, root in (("original_val", args.original_val), ("jic", args.jic_val)):
        print(f"Scoring {name}...", flush=True)
        result["sources"][name] = evaluate(name, root, model, args.output_dir, args.batch_size)

    expected = {"original_val": 728, "jic": 188}
    result["historical_reproduction"] = {}
    for name, correct in expected.items():
        measured = result["sources"][name]["classification"]["correct"]
        result["historical_reproduction"][name] = {
            "expected_correct": correct, "observed_correct": measured,
            "matches": measured == correct,
        }
    if not all(entry["matches"] for entry in result["historical_reproduction"].values()):
        result["warnings"].append(
            "A baseline did not match published results: investigate model/data versions "
            "and preprocessing before interpreting confidence metrics."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "reliability_report.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "correct": {n: v["classification"]["correct"] for n, v in result["sources"].items()},
        "matches_historical": {n: v["matches"] for n, v in result["historical_reproduction"].items()},
        "high_confidence_mistakes_at_0_9": {
            n: v["mistakes_with_score_at_least_0_9"] for n, v in result["sources"].items()
        },
        "outputs": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
