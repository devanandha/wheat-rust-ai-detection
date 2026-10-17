"""Run one prespecified Wheat Rust AI dataset-size/source-diversity condition.

Self-contained model and dataset loaders intentionally avoid dependency on older
versions of src.model.build_classifier and src.data.load_split in local clones.
This runner preserves original model scripts/checkpoints and keeps JIC benchmark
out of training and checkpoint selection.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

CLASSES = ["Brown_rust", "Healthy", "Yellow_rust"]
IMAGE_SIZE = (224, 224)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for part in iter(lambda: file.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def load_images(folder, *, shuffle, batch_size, seed):
    """Return tf.data.Dataset and stable filenames for nonshuffled evaluation."""
    if not Path(folder).is_dir():
        raise FileNotFoundError(f"Dataset directory missing: {folder}")
    ds = tf.keras.utils.image_dataset_from_directory(
        str(folder), labels="inferred", label_mode="categorical",
        class_names=CLASSES, image_size=IMAGE_SIZE, batch_size=batch_size,
        shuffle=shuffle, seed=seed if shuffle else None,
    )
    paths = tuple(ds.file_paths) if not shuffle else None
    return ds.prefetch(tf.data.AUTOTUNE), paths


def build_study_model(architecture):
    """Architecture-matched frozen-backbone transfer-learning study model.

    Matches the CURRENT public repository's classifier recipe, regardless of the
    local historical src.model.py version. Does not modify historical models.
    """
    if architecture == "mobilenetv2":
        backbone = tf.keras.applications.MobileNetV2(
            input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet")
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    elif architecture == "efficientnetb0":
        backbone = tf.keras.applications.EfficientNetB0(
            input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet")
        preprocess = tf.keras.applications.efficientnet.preprocess_input
    elif architecture == "resnet50":
        backbone = tf.keras.applications.ResNet50(
            input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet")
        preprocess = tf.keras.applications.resnet50.preprocess_input
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    backbone.trainable = False
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="image")
    x = tf.keras.layers.RandomFlip("horizontal")(inputs)
    x = tf.keras.layers.RandomRotation(0.08)(x)
    x = tf.keras.layers.RandomZoom(0.10)(x)
    x = tf.keras.layers.RandomContrast(0.10)(x)
    x = preprocess(x)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    output = tf.keras.layers.Dense(len(CLASSES), activation="softmax", name="prediction")(x)
    model = tf.keras.Model(inputs, output, name=f"wheat_rust_{architecture}_extension")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def evaluate_split(model, folder, output_folder, batch_size, seed):
    ds, paths = load_images(folder, shuffle=False, batch_size=batch_size, seed=seed)
    actual = np.concatenate([np.argmax(batch_labels.numpy(), axis=1)
                             for _, batch_labels in ds])
    probabilities = model.predict(ds, verbose=0)
    if len(paths) != len(actual) or len(probabilities) != len(actual):
        raise RuntimeError("Image path and evaluation order mismatch")
    predictions = np.argmax(probabilities, axis=1)
    output_folder.mkdir(parents=True, exist_ok=False)
    with (output_folder / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "actual", "predicted",
                         *[f"prob_{name}" for name in CLASSES]])
        for path, a, p, proba in zip(paths, actual, predictions, probabilities):
            writer.writerow([Path(path).name, CLASSES[int(a)], CLASSES[int(p)],
                             *[float(v) for v in proba]])
    report = classification_report(actual, predictions, labels=list(range(len(CLASSES))),
                                   target_names=CLASSES, output_dict=True, zero_division=0)
    matrix = confusion_matrix(actual, predictions, labels=list(range(len(CLASSES))))
    (output_folder / "classification_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    np.savetxt(output_folder / "confusion_matrix.csv", matrix, fmt="%d", delimiter=",")
    return {"images": int(len(actual)), "accuracy": float(report["accuracy"]),
            "macro_f1": float(report["macro avg"]["f1-score"])}


def run(args):
    if not args.manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("overlap_screen") != "SHA-256 and ImageHash pHash":
        raise ValueError("This run requires the prepared SHA-256 + pHash overlap screen")
    entry = next((x for x in manifest["conditions"] if x["id"] == args.condition), None)
    if entry is None:
        raise ValueError(f"Unknown condition {args.condition}")
    root = args.manifest.resolve().parent
    output = root / entry["id"] / "results"
    if output.exists():
        raise FileExistsError(f"Results folder exists; do not overwrite previous run: {output}")
    tf.keras.utils.set_random_seed(int(entry["seed"]))
    train, _ = load_images(Path(entry["train_dir"]), shuffle=True,
                           batch_size=args.batch_size, seed=int(entry["seed"]))
    validation, _ = load_images(Path(manifest["internal_validation"]), shuffle=False,
                                batch_size=args.batch_size, seed=int(entry["seed"]))
    model = build_study_model(args.architecture)
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = output / "best_model.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.3),
    ]
    history = model.fit(train, validation_data=validation, epochs=args.epochs,
                        callbacks=callbacks, verbose=1)
    (output / "training_history.json").write_text(
        json.dumps(history.history, indent=2), encoding="utf-8")
    checkpoint_model = tf.keras.models.load_model(checkpoint)
    internal = evaluate_split(checkpoint_model, Path(manifest["internal_validation"]),
                              output / "internal", args.batch_size, int(entry["seed"]))
    # The fixed JIC benchmark is evaluated ONLY AFTER checkpoint choice.
    external = evaluate_split(checkpoint_model, Path(manifest["external_heldout_test"]),
                              output / "external", args.batch_size, int(entry["seed"]))
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    metadata = {
        "condition": entry["id"], "architecture": args.architecture,
        "model_recipe": "standalone-extension-v3; frozen ImageNet backbone; augmentation",
        "seed": int(entry["seed"]), "epochs_max": args.epochs,
        "batch_size": args.batch_size, "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "tensorflow_version": tf.__version__, "numpy_version": np.__version__,
        "git_commit": commit, "runner_sha256": sha256(Path(__file__)),
        "manifest_sha256": sha256(args.manifest), "best_model_sha256": sha256(checkpoint),
        "internal": internal, "external": external,
        "external_used_for_training_or_checkpoint_selection": False,
        "external_benchmark_previously_evaluated": True,
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path("experiments/generalisation/study_manifest.json"))
    parser.add_argument("--condition", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--architecture", choices=("mobilenetv2", "efficientnetb0", "resnet50"),
                        default="mobilenetv2")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch size must be positive")
    run(args)


if __name__ == "__main__":
    main()
