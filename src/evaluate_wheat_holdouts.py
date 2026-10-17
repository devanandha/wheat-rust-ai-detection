"""Read-only evaluation of Wheat Rust AI saved classifiers on protected source holdouts.

Outputs are written under a separate directory; input datasets and training results
are never modified. Multi-model mode launches each model in a fresh Python process
to release TensorFlow RAM on memory-constrained computers.

Each holdout has only two of the trained classifier's three categories. We report
accuracy and macro-F1 for PRESENT true classes, while retaining a full 3x3
confusion matrix so predictions of an absent class are still penalized and visible.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

CLASSES = ("Brown_rust", "Healthy", "Yellow_rust")
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".gif"})
DATASETS = {
    "iari_test": {"control": "Healthy", "diseased": "Brown_rust"},
    "ethiopia_holdout": {"Healthy": "Healthy", "Yellow_rust": "Yellow_rust"},
}
EXPECTED_COUNTS = {
    "iari_test": {"Brown_rust": 55, "Healthy": 74},
    "ethiopia_holdout": {"Healthy": 20, "Yellow_rust": 41},
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_holdout(root, dataset):
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Missing holdout folder: {root}")
    mapping = DATASETS[dataset]
    actual_dirs = {p.name for p in root.iterdir() if p.is_dir()}
    if actual_dirs != set(mapping):
        raise ValueError(f"Unexpected {dataset} class folders: {sorted(actual_dirs)}. Expected {sorted(mapping)}")
    examples = []
    for directory, destination in mapping.items():
        folder = root / directory
        for path in sorted(folder.rglob("*"), key=lambda p: p.as_posix().lower()):
            if path.is_file():
                if path.suffix.lower() not in IMAGE_EXTENSIONS:
                    raise ValueError(f"Unexpected non-image in holdout: {path}")
                examples.append((path, destination))
    examples.sort(key=lambda pair: pair[0].relative_to(root).as_posix().lower())
    counts = {c: sum(label == c for _, label in examples) for c in CLASSES}
    found = {k: v for k, v in counts.items() if v}
    if found != EXPECTED_COUNTS[dataset]:
        raise ValueError(f"Holdout {dataset} counts changed: {found}; expected {EXPECTED_COUNTS[dataset]}")
    return root, examples, found


def checkpoint_info(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file() or checkpoint.name != "best_model.keras":
        raise FileNotFoundError(f"Checkpoint must be an existing best_model.keras: {checkpoint}")
    if checkpoint.parent.name != "results":
        raise ValueError(f"Require checkpoint in the original results folder: {checkpoint}")
    condition = checkpoint.parent.parent.name
    group = checkpoint.parent.parent.parent.name
    metadata_file = checkpoint.parent / "run_metadata.json"
    if not metadata_file.is_file():
        raise FileNotFoundError(f"Missing run metadata: {metadata_file}")
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    if metadata.get("condition") != condition:
        raise ValueError(f"Metadata condition mismatch at {checkpoint}")
    if metadata.get("architecture") != "mobilenetv2":
        raise ValueError(f"Unexpected architecture at {checkpoint}: {metadata.get('architecture')}")
    expected_sha = metadata.get("best_model_sha256")
    current_sha = digest(checkpoint)
    if expected_sha != current_sha:
        raise ValueError(f"Saved checkpoint hash does not match training metadata: {checkpoint}")
    return {"checkpoint": checkpoint, "condition": condition, "group": group,
            "metadata": metadata, "model_sha256": current_sha}


def discover(roots):
    entries = []
    seen = set()
    for supplied_root in roots:
        root = Path(supplied_root).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Experiment root not found: {root}")
        for checkpoint in sorted(root.rglob("best_model.keras")):
            if checkpoint.parent.name != "results":
                continue  # deliberately exclude archived failed attempts
            info = checkpoint_info(checkpoint)
            if str(info["checkpoint"]) not in seen:
                seen.add(str(info["checkpoint"]))
                entries.append(info)
    names = [(e["group"], e["condition"]) for e in entries]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate group/condition output names detected")
    return sorted(entries, key=lambda x: (x["group"], x["condition"]))


def evaluate_one(info, datasets, output_root, batch_size):
    # Only worker imports TensorFlow, keeping the coordinator process lightweight.
    import numpy as np
    import tensorflow as tf
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score

    output = Path(output_root).resolve() / info["group"] / info["condition"]
    final_report = output / "summary.json"
    if output.exists():
        if final_report.is_file():
            old = json.loads(final_report.read_text(encoding="utf-8"))
            if old.get("model_sha256") == info["model_sha256"]:
                raise FileExistsError(f"Evaluation already exists (preserved): {final_report}")
        raise FileExistsError(f"Partial/different evaluation output exists; preserve/rename it first: {output}")
    # Make sure data and model can be loaded before creating the output directory.
    for name, root in datasets.items():
        scan_holdout(root, name)
    tf.keras.utils.set_random_seed(int(info["metadata"]["seed"]))
    model = tf.keras.models.load_model(str(info["checkpoint"]), compile=False)
    if len(model.output_shape) != 2 or model.output_shape[-1] != 3:
        raise ValueError(f"Unexpected model output shape {model.output_shape}")
    if tuple(model.input_shape[1:]) != (224, 224, 3):
        raise ValueError(f"Unexpected model input shape {model.input_shape}")

    output.mkdir(parents=True, exist_ok=False)
    per_dataset = {}
    for name, supplied_root in datasets.items():
        root, examples, class_counts = scan_holdout(supplied_root, name)
        files = [str(p) for p, _ in examples]
        y_true = np.asarray([CLASSES.index(c) for _, c in examples], dtype=np.int64)

        # Equivalent decoded-image / bilinear-resize input to Keras
        # image_dataset_from_directory(image_size=(224,224)), which returns
        # unscaled float RGB pixels. MobileNetV2 preprocessing resides in model.
        def decode_and_resize(path):
            raw = tf.io.read_file(path)
            image = tf.io.decode_image(raw, channels=3, expand_animations=False)
            image.set_shape((None, None, 3))
            image = tf.image.resize(image, [224, 224], method="bilinear")
            return tf.cast(image, tf.float32)

        ds = (tf.data.Dataset.from_tensor_slices(files)
              .map(decode_and_resize, num_parallel_calls=1)
              .batch(batch_size).prefetch(1))
        probabilities = model.predict(ds, verbose=0)
        if len(probabilities) != len(y_true) or probabilities.shape[1] != 3:
            raise RuntimeError(f"Prediction count/shape mismatch for {name}: {probabilities.shape}")
        y_pred = np.argmax(probabilities, axis=1)
        present = sorted(set(y_true.tolist()))
        missing = sorted(set(range(3)) - set(present))
        full_report = classification_report(y_true, y_pred, labels=[0, 1, 2],
                                            target_names=list(CLASSES), output_dict=True, zero_division=0)
        # Mean over actually represented classes, not an arbitrary zero from
        # classes that the test set does not contain.
        macro_f1_present = float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0))
        macro_recall_present = float(recall_score(y_true, y_pred, labels=present, average="macro", zero_division=0))
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        out = output / name
        out.mkdir()
        with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["relative_file", "actual", "predicted", *("prob_" + c for c in CLASSES)])
            for (path, label), pred, prob in zip(examples, y_pred, probabilities):
                writer.writerow([path.relative_to(root).as_posix(), label, CLASSES[int(pred)],
                                 *(float(x) for x in prob)])
        (out / "classification_report.json").write_text(json.dumps(full_report, indent=2), encoding="utf-8")
        with (out / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["actual\\predicted", *CLASSES])
            for label, values in zip(CLASSES, cm.tolist()):
                writer.writerow([label, *values])
        absent_prediction_count = int(sum(int(k in missing) for k in y_pred))
        results = {
            "images": len(examples), "actual_class_counts": class_counts,
            "present_classes": [CLASSES[i] for i in present],
            "absent_classes": [CLASSES[i] for i in missing],
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1_present_classes": macro_f1_present,
            "balanced_accuracy_present_classes": macro_recall_present,
            "predictions_of_absent_class": absent_prediction_count,
            "predicted_class_counts": {cls: int(sum(y_pred == i)) for i, cls in enumerate(CLASSES)},
            "full_confusion_matrix": cm.tolist(),
        }
        per_dataset[name] = results
        print(f"{name}: {len(examples)} images, accuracy {results['accuracy']:.4f}, "
              f"present-class macro-F1 {macro_f1_present:.4f}, "
              f"absent-class predictions {absent_prediction_count}", flush=True)

    summary = {
        "condition": info["condition"], "experiment_group": info["group"],
        "architecture": "mobilenetv2", "seed": info["metadata"]["seed"],
        "model_sha256": info["model_sha256"], "checkpoint_path": str(info["checkpoint"]),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_protocol": "separate source holdouts; three-class classifier, two observed classes per holdout",
        "checkpoint_selected_using_holdouts": False,
        "jic_used_for_checkpoint_selection": False,
        "provenance_independence_not_proven": True,
        "classes": list(CLASSES), "datasets": per_dataset,
    }
    final_report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved: {final_report}", flush=True)
    return summary


def aggregate_report(output_root, infos):
    rows = []
    for info in infos:
        file = Path(output_root).resolve() / info["group"] / info["condition"] / "summary.json"
        if not file.is_file():
            raise RuntimeError(f"Missing final evaluation report: {file}")
        summary = json.loads(file.read_text(encoding="utf-8"))
        if summary.get("model_sha256") != info["model_sha256"]:
            raise ValueError(f"Model SHA mismatch in {file}")
        row = {"experiment_group": info["group"], "condition": info["condition"],
               "seed": summary["seed"]}
        for dataset in DATASETS:
            record = summary["datasets"][dataset]
            for key in ("images", "accuracy", "macro_f1_present_classes",
                        "balanced_accuracy_present_classes", "predictions_of_absent_class"):
                row[f"{dataset}_{key}"] = record[key]
        rows.append(row)
    output = Path(output_root).resolve() / "all_holdout_results.csv"
    if output.exists():
        raise FileExistsError(f"Aggregated report already exists; will not overwrite: {output}")
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCompleted {len(rows)} models. Consolidated CSV: {output}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint", type=Path, help="Evaluate one saved best_model.keras checkpoint")
    g.add_argument("--model-roots", type=Path, nargs="+", help="Evaluate all checkpoint folders in these experiment roots")
    p.add_argument("--iari-test", type=Path, required=True)
    p.add_argument("--ethiopia-holdout", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=8, help="Evaluation-only batch size; model weights unchanged")
    p.add_argument("--expected-models", type=int, default=18, help="Expected number of models in multi-model mode")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.batch_size < 1:
        p.error("--batch-size must be >=1")
    datasets = {"iari_test": args.iari_test, "ethiopia_holdout": args.ethiopia_holdout}
    for name, root in datasets.items():
        _, _, counts = scan_holdout(root, name)
        print(f"Verified {name}: {counts}")

    if args.checkpoint:
        info = checkpoint_info(args.checkpoint)
        print(f"Evaluating {info['group']} / {info['condition']}", flush=True)
        evaluate_one(info, datasets, args.out_dir, args.batch_size)
        return

    infos = discover(args.model_roots)
    if len(infos) != args.expected_models:
        print("Discovered candidate checkpoints:")
        for info in infos:
            print(f"  {info['group']} / {info['condition']}: {info['checkpoint']}")
        raise RuntimeError(f"Expected {args.expected_models} completed checkpoints, found {len(infos)}. No evaluations started.")
    print(f"Verified {len(infos)} completed checkpoints", flush=True)
    # One fresh process per checkpoint avoids TensorFlow memory retention across 18 models.
    for i, info in enumerate(infos, start=1):
        report = args.out_dir.resolve() / info["group"] / info["condition"] / "summary.json"
        if report.is_file():
            old = json.loads(report.read_text(encoding="utf-8"))
            if old.get("model_sha256") == info["model_sha256"]:
                print(f"[{i}/{len(infos)}] Skipping completed {info['group']} / {info['condition']}", flush=True)
                continue
            raise ValueError(f"Previous evaluation uses a different checkpoint: {report}")
        print(f"\n[{i}/{len(infos)}] Evaluating {info['group']} / {info['condition']}", flush=True)
        command = [sys.executable, str(Path(__file__).resolve()),
                   "--checkpoint", str(info["checkpoint"]),
                   "--iari-test", str(args.iari_test.resolve()),
                   "--ethiopia-holdout", str(args.ethiopia_holdout.resolve()),
                   "--out-dir", str(args.out_dir.resolve()),
                   "--batch-size", str(args.batch_size), "--worker"]
        subprocess.run(command, check=True, env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "2"})
    aggregate_report(args.out_dir, infos)


if __name__ == "__main__":
    main()
