"""Isolated one-image inference worker for the local field-research app.

Reads raw image bytes from standard input and prints one result line on stdout.
Runs TensorFlow only in this short-lived process so the Streamlit UI need not
load TensorFlow, sklearn or scipy into its own memory. No photo files are saved.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

# Set these before importing TensorFlow. Restrict CPU-thread pressure.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

from src.field_research import MAX_FILE_BYTES, inspect_photo, sha256_file


RESULT_PREFIX = "FIELD_RESULT_JSON:"


def run_inference(model_path: Path, raw_photo: bytes) -> dict:
    if not model_path.is_file():
        raise ValueError(f"Model path not found: {model_path}")
    photo, metrics = inspect_photo(raw_photo)
    # Import TensorFlow ONLY in this short-lived child process.
    import numpy as np
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path, compile=False)
    pixels = tf.convert_to_tensor(np.asarray(photo, dtype=np.float32))
    # Match original src.data loader: RGB 0..255 bilinear 224, no extra
    # external MobileNetV2 preprocessing (it lives inside the model).
    resized = tf.image.resize(pixels, (224, 224), method="bilinear")
    scores = np.asarray(
        model(tf.expand_dims(resized, 0), training=False), dtype=np.float64
    )
    if scores.shape != (1, 3) or not np.all(np.isfinite(scores)):
        raise ValueError(f"Unexpected model output: {scores.shape}")
    values = list(map(float, scores[0]))
    if np.any(scores < 0) or np.any(scores > 1) or not np.isclose(sum(values), 1, atol=1e-3):
        raise ValueError("Unusable class probabilities from saved model.")
    return {
        "scores": values,
        "model_sha256": sha256_file(model_path),
        "width_px": metrics["width_px"],
        "height_px": metrics["height_px"],
        "privacy": "no_original_image_written",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    args = parser.parse_args()
    # Only one permitted image; cap even a malicious or accidental large stdin.
    raw = sys.stdin.buffer.read(MAX_FILE_BYTES + 1)
    result = run_inference(args.model, raw)
    print(RESULT_PREFIX + json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
