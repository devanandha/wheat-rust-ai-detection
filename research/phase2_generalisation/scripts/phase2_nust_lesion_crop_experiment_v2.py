"""Experiment 7: exploratory NUST lesion-centred crop diagnostic.

Run from repository root:
    python research/phase2_generalisation/scripts/phase2_nust_lesion_crop_experiment_v2.py

Mask-informed crops use ground-truth information unavailable at deployment.
This is a retrospective diagnostic analysis, not a preregistered trial
or an unbiased estimate of real-world performance.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

CLASSES = ["Brown_rust", "Healthy", "Yellow_rust"]
from phase2_config import NUST_REFERENCE as CSV, MODEL, OUT as OUTBASE, nust_path
OUT = OUTBASE / 'phase2_nust_crop_experiment'
SIZE = 224
CROP_FRACTION = 0.50  # Square crop occupies half the image's shorter dimension.
GRID_STEPS = 9  # Prespecified 9 x 9 search positions for BOTH selection methods.


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate_boxes(width: int, height: int, side: int):
    xs = np.rint(np.linspace(0, width - side, GRID_STEPS)).astype(int)
    ys = np.rint(np.linspace(0, height - side, GRID_STEPS)).astype(int)
    return [(int(x), int(y), int(x + side), int(y + side)) for y in ys for x in xs]


def box_lesion_pixels(mask: Image.Image, box: tuple[int, int, int, int]) -> int:
    return int(np.count_nonzero(np.asarray(mask.crop(box)) > 127))


def select_boxes(mask: Image.Image, width: int, height: int):
    side = max(1, int(round(min(width, height) * CROP_FRACTION)))
    boxes = candidate_boxes(width, height, side)
    # Work with a small nearest-neighbour proxy to choose crop locations cheaply.
    # We separately measure actual lesion counts on the chosen full-res masks.
    proxy = np.asarray(mask.resize((256, 256), Image.Resampling.NEAREST)) > 127
    integral = np.pad(proxy.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    vals = []
    for x0, y0, x1, y1 in boxes:
        ax, bx = int(round(x0 * 256 / width)), int(round(x1 * 256 / width))
        ay, by = int(round(y0 * 256 / height)), int(round(y1 * 256 / height))
        vals.append(int(integral[by, bx] - integral[ay, bx] - integral[by, ax] + integral[ay, ax]))
    max_idx = int(np.argmax(vals))
    min_idx = int(np.argmin(vals))
    return boxes[max_idx], boxes[min_idx], vals[max_idx] == vals[min_idx]


def resized_rgb_from_image(image_tensor: tf.Tensor, box=None):
    if box is not None:
        x0, y0, x1, y1 = box
        image_tensor = tf.image.crop_to_bounding_box(image_tensor, y0, x0, y1 - y0, x1 - x0)
    return tf.image.resize(image_tensor, (SIZE, SIZE), method="bilinear")


def exact_mcnemar_p(a: list[bool], b: list[bool]) -> dict:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    gain = int(np.sum(~a & b))
    loss = int(np.sum(a & ~b))
    discordant = gain + loss
    if discordant == 0:
        p_value = 1.0
    else:
        p_value = min(1., 2 * sum(math.comb(discordant, k) for k in range(min(gain, loss) + 1)) / 2**discordant)
    return {"wrong_to_yellow": gain, "yellow_to_wrong": loss, "exact_mcnemar_two_sided_p": p_value}


def summarize(df: pd.DataFrame, label: str):
    subset = df if label == "all" else df[df["split"] == label]
    results = {"n": len(subset)}
    for condition in ("full", "lesion", "control"):
        predictions = subset[f"{condition}_predicted_label"]
        detected = int((predictions == "Yellow_rust").sum())
        results[condition] = {
            "yellow_predictions": detected,
            "yellow_detection_fraction_conditional_on_labels": detected / len(subset),
            "prediction_counts": {c: int((predictions == c).sum()) for c in CLASSES},
            "mean_yellow_score": float(subset[f"{condition}_prob_yellow_rust"].mean()),
        }
    for first, second in (("full", "lesion"), ("full", "control"), ("control", "lesion")):
        results[f"{first}_vs_{second}"] = exact_mcnemar_p(
            (subset[f"{first}_predicted_label"] == "Yellow_rust").tolist(),
            (subset[f"{second}_predicted_label"] == "Yellow_rust").tolist(),
        )
    return results


def main():
    if not CSV.is_file():
        raise FileNotFoundError(f"Missing existing NUST predictions: {CSV}")
    if not MODEL.is_file():
        raise FileNotFoundError(f"Missing saved model: {MODEL}")
    source = pd.read_csv(CSV)
    needed = {"split", "image_name", "image_path", "mask_path", "image_sha256", "predicted_label",
              "prob_brown_rust", "prob_healthy", "prob_yellow_rust", "assumed_true_label"}
    if not needed.issubset(source.columns):
        raise ValueError(f"Missing CSV columns: {needed - set(source.columns)}")
    if len(source) != 100 or source["image_path"].duplicated().any():
        raise ValueError("Expected exactly 100 unique image rows; refusing partial/duplicate evaluation")
    if not (source["assumed_true_label"] == "Yellow_rust").all():
        raise ValueError("Unexpected NUST label interpretation")
    print("Experiment 7: paired 50%-short-side square crops on 9x9 grid")
    print("FULL = original image; LESION = highest mask area; CONTROL = lowest mask area")
    print("Mask-informed selection is diagnostic only; the model is never trained.")
    tf.keras.utils.set_random_seed(42)
    model = tf.keras.models.load_model(MODEL, compile=False)
    rows = []
    for i, original in enumerate(source.itertuples(index=False), 1):
        image_path = nust_path(original.image_path)
        mask_path = nust_path(original.mask_path)
        if sha256(image_path) != original.image_sha256:
            raise ValueError(f"Image changed since previous evaluation: {image_path}")
        raw = tf.io.read_file(str(image_path))
        image = tf.io.decode_jpeg(raw, channels=3)
        h, w = int(image.shape[0]), int(image.shape[1])
        with Image.open(mask_path) as source_mask:
            mask = source_mask.convert("L")
            original_mask_size = mask.size
            mask_resized_to_image = original_mask_size != (w, h)
            if mask_resized_to_image:
                mw, mh = original_mask_size
                mismatch = max(abs(mw - w) / w, abs(mh - h) / h)
                # Preserve masks unless a minor resolution discrepancy can be reconciled.
                # This assumes corresponding full-image framing, NOT verified registration.
                if mismatch > 0.02:
                    raise ValueError(
                        f"Mask mismatch exceeds 2%: {mask_path} "
                        f"mask={original_mask_size} image={(w, h)}; manual alignment review required"
                    )
                mask = mask.resize((w, h), Image.Resampling.NEAREST)
                print(f"WARNING: Rescaled mask with nearest-neighbour for {original.image_name}: "
                      f"{original_mask_size} -> {(w, h)}; verify registration visually")
            lesion_box, control_box, grid_tie = select_boxes(mask, w, h)
            lesion_pixels = box_lesion_pixels(mask, lesion_box)
            control_pixels = box_lesion_pixels(mask, control_box)
        # Proxy selection is approximate; retain and report occasional inversions instead of changing selection after the fact.
        # Exactly the same full-image preprocessing as saved baseline:
        # JPEG decode -> RGB -> tf.image.resize bilinear 224x224 float32 [0,255].
        # MobileNetV2 preprocessing is embedded in the saved model.
        batch = tf.stack([
            resized_rgb_from_image(image),
            resized_rgb_from_image(image, lesion_box),
            resized_rgb_from_image(image, control_box),
        ])
        scores = np.asarray(model(batch, training=False).numpy(), dtype=float)
        baseline = np.array([original.prob_brown_rust, original.prob_healthy, original.prob_yellow_rust], dtype=float)
        if not np.allclose(scores[0], baseline, atol=2e-4, rtol=2e-4):
            raise RuntimeError(
                f"Baseline mismatch at {image_path}: current={scores[0].tolist()}, saved={baseline.tolist()}. "
                "Stop: inspect input preprocessing/model before interpreting any crop results."
            )
        row = {"split": original.split, "image_name": original.image_name,
               "image_path": str(original.image_path).replace(chr(92), "/"), "mask_path": str(original.mask_path).replace(chr(92), "/"),
               "mask_original_width": original_mask_size[0], "mask_original_height": original_mask_size[1],
               "mask_resized_to_image": mask_resized_to_image,
               "lesion_box_xyxy": json.dumps(lesion_box), "control_box_xyxy": json.dumps(control_box),
               "lesion_crop_mask_pixels": lesion_pixels, "control_crop_mask_pixels": control_pixels,
               "equal_proxy_density_flag": bool(grid_tie),
               "proxy_order_inversion_flag": bool(lesion_pixels < control_pixels)}
        for tag, score in zip(("full", "lesion", "control"), scores):
            row[f"{tag}_predicted_label"] = CLASSES[int(np.argmax(score))]
            row[f"{tag}_confidence"] = float(score.max())
            for cls, value in zip(CLASSES, score):
                row[f"{tag}_prob_{cls.lower()}"] = float(value)
        rows.append(row)
        if i % 10 == 0:
            print(f"Verified baseline and evaluated {i}/100")
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    output_csv = OUT / "paired_nust_lesion_crops.csv"
    output_json = OUT / "summary.json"
    df.to_csv(output_csv, index=False)
    summary = {
        "experiment": "NUST frozen model diagnostic, 50%-short-side square crops, 9x9 position grid",
        "model_sha256": sha256(MODEL), "source_predictions_sha256": sha256(CSV),
        "class_order": CLASSES, "crop_fraction": CROP_FRACTION, "grid_steps_per_axis": GRID_STEPS,
        "ground_truth_assumption": "All 100 images labelled yellow from nonempty binary rust-lesion masks; verify source definitions",
        "caveats": ["Uses ground-truth segmentation masks to select crops; NOT deployment performance",
                    "Control crop is least lesion-dense among matching grid crops, not a random independent control",
                    "Cropping changes symptom visibility, scale, scene context and background simultaneously",
                    "Perceptual similarity checks do not establish complete absence of source overlap",
                    "Near-size-matched masks are nearest-neighbour resized; spatial registration must be visually checked before causal interpretation"],
        "counts": {"rescaled_masks": int(df["mask_resized_to_image"].sum()),
                   "mask_grid_ties": int(df["equal_proxy_density_flag"].sum()),
                   "equal_actual_mask_counts": int((df["lesion_crop_mask_pixels"] == df["control_crop_mask_pixels"]).sum()),
                   "proxy_order_inversions": int(df["proxy_order_inversion_flag"].sum())},
        "all": summarize(df, "all"), "train": summarize(df, "train"), "test": summarize(df, "test")
    }
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSUMMARY")
    print(json.dumps(summary["all"], indent=2))
    print("\nSaved:", output_csv)
    print("Saved:", output_json)


if __name__ == "__main__":
    main()
