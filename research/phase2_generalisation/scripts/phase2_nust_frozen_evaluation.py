"""Frozen-model NUST yellow-rust image-level evaluation.

Usage (from repository root): python phase2_nust_frozen_evaluation.py
Requires TensorFlow, Pillow, NumPy. Does not modify datasets or model.
Assumes NWRD annotation documentation confirms positive masks mark stripe-rust lesions.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf

from phase2_config import NUST_DIR, MODEL as MODEL_PATH, OUT as OUTBASE
OUTPUT_DIR = OUTBASE / 'phase2_nust_evaluation'
CLASSES = ("Brown_rust", "Healthy", "Yellow_rust")
IMAGE_SIZE = (224, 224)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054):
    if not total:
        return None
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [round(max(0.0, center - margin), 6), round(min(1.0, center + margin), 6)]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def image_to_tensor(path: Path):
    # Mirrors image_dataset_from_directory's decoded RGB, 224x224 bilinear resize,
    # unnormalized float pixels. Model already contains MobileNetV2 preprocess_input.
    pixels = tf.io.decode_jpeg(tf.io.read_file(str(path)), channels=3)
    pixels = tf.image.resize(pixels, IMAGE_SIZE, method="bilinear")
    return tf.expand_dims(tf.cast(pixels, tf.float32), axis=0)


def prepare_manifest():
    rows = []
    for split in ("train", "test"):
        image_dir = NUST_DIR / split / "images"
        mask_dir = NUST_DIR / split / "masks"
        if not image_dir.is_dir() or not mask_dir.is_dir():
            raise FileNotFoundError(f"Missing NUST split folders: {image_dir} or {mask_dir}")
        imgs = sorted(image_dir.glob("*.jpg"), key=lambda f: f.name)
        masks = sorted(mask_dir.glob("*.png"), key=lambda f: f.name)
        img_stems = {f.stem for f in imgs}
        mask_stems = {f.stem for f in masks}
        if len(imgs) != len(img_stems) or len(masks) != len(mask_stems):
            raise ValueError(f"Duplicate basename in {split} files")
        if img_stems != mask_stems:
            raise ValueError(f"Unpaired images or masks in {split}: images-only={sorted(img_stems - mask_stems)}, masks-only={sorted(mask_stems - img_stems)}")
        for img in imgs:
            mask = mask_dir / (img.stem + ".png")
            with Image.open(mask) as im:
                mask_np = np.asarray(im.convert("L"))
            positive_pixels = int(np.count_nonzero(mask_np > 127))
            if positive_pixels == 0:
                raise ValueError(f"Unexpected empty lesion mask: {mask}")
            rows.append({
                "split": split,
                "image_name": img.name,
                "image_path": str(img),
                "mask_path": str(mask),
                "image_sha256": sha256(img),
                "mask_positive_pixels": positive_pixels,
                "mask_total_pixels": int(mask_np.size),
                "lesion_coverage_pct": round(100 * positive_pixels / mask_np.size, 6),
                "assumed_true_label": "Yellow_rust",
            })
    if len(rows) != 100:
        raise ValueError(f"Expected 100 NUST images, found {len(rows)}. Refusing to silently change cohort.")
    return rows


def summarize(rows):
    n = len(rows)
    correct = sum(r["predicted_label"] == "Yellow_rust" for r in rows)
    return {
        "n": n,
        "yellow_rust_predicted": correct,
        "yellow_rust_recall_if_all_positive": round(correct / n, 6) if n else None,
        "yellow_rust_recall_wilson_95pct_ci": wilson_interval(correct, n),
        "predicted_class_counts": dict(Counter(r["predicted_label"] for r in rows)),
        "high_confidence_wrong_at_0_9": sum(r["predicted_label"] != "Yellow_rust" and r["confidence"] >= 0.9 for r in rows),
        "mean_yellow_probability": round(float(np.mean([r["prob_yellow_rust"] for r in rows])), 6) if n else None,
    }


def main():
    print("NUST independent-source frozen-model evaluation")
    print("Model:", MODEL_PATH)
    print("Dataset:", NUST_DIR)
    print("IMPORTANT: All 100 image-level positive labels assume lesion masks annotate yellow/stripe rust.")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Saved classifier not found: {MODEL_PATH}")
    manifest = prepare_manifest()
    print("Validated image-mask pairs:", len(manifest))
    print("Loading frozen saved model; no training or weight updates...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    if model.output_shape[-1] != 3 or model.input_shape[-3:] != (224, 224, 3):
        raise ValueError(f"Unexpected saved model shapes: input {model.input_shape}, output {model.output_shape}")
    results = []
    for idx, r in enumerate(manifest, 1):
        probs = np.asarray(model(image_to_tensor(Path(r["image_path"])), training=False))[0]
        if len(probs) != 3 or not np.all(np.isfinite(probs)):
            raise ValueError(f"Invalid output probabilities for {r['image_path']}: {probs}")
        pred_idx = int(np.argmax(probs))
        results.append({**r,
            "predicted_label": CLASSES[pred_idx],
            "confidence": round(float(probs[pred_idx]), 8),
            "prob_brown_rust": round(float(probs[0]), 8),
            "prob_healthy": round(float(probs[1]), 8),
            "prob_yellow_rust": round(float(probs[2]), 8),
        })
        if idx % 10 == 0:
            print(f"Evaluated {idx}/100")
    summary = {
        "evaluation_type": "new-source, frozen-model, one-class (yellow/stripe rust) image-level evaluation",
        "model_file": str(MODEL_PATH),
        "model_sha256": sha256(MODEL_PATH),
        "dataset_root": str(NUST_DIR),
        "class_order": CLASSES,
        "preprocessing": "tf.io.decode_jpeg RGB, tf.image.resize bilinear 224x224 float32 0..255; MobileNetV2 preprocessing embedded in model",
        "ground_truth_caveat": "Positive image-level label is inferred from nonempty segmentation masks, conditional on dataset authors defining foreground as yellow-rust lesions; no healthy or brown-rust reference labels exist in this test.",
        "whole_cohort": summarize(results),
        "by_original_split": {
            "train": summarize([r for r in results if r["split"] == "train"]),
            "test": summarize([r for r in results if r["split"] == "test"]),
        },
        "limitations": [
            "One-class dataset: recall only, not three-class accuracy, specificity, or generalisation across all classes.",
            "Train/test refers to original NUST segmentation split; neither split was used to train this classifier.",
            "Image and scene shifts, crop/recompression overlap, and segmentation-to-classification label differences remain possible.",
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTPUT_DIR / "nust_frozen_predictions.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    out_json = OUTPUT_DIR / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSUMMARY")
    print(json.dumps({"whole_cohort": summary["whole_cohort"], "by_original_split": summary["by_original_split"]}, indent=2))
    print("\nSaved:", out_csv)
    print("Saved:", out_json)


if __name__ == "__main__":
    main()
