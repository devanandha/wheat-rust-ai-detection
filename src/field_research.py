"""Local-only helpers for the Wheat Rust AI field-research prototype.

None of these image-quality heuristics is a validated disease/leaf detector.
Only metadata and model/reviewer outputs are logged; image bytes are never saved.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import uuid

import numpy as np
from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError

MODEL_CLASSES = ("Brown_rust", "Healthy", "Yellow_rust")
REFERENCE_LABELS = (*MODEL_CLASSES, "Other_or_mixed", "Uncertain_unconfirmed")
MAX_FILE_BYTES = 12 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
VERSION = "field-prototype-0.1"


def sha256_bytes(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_photo(contents: bytes) -> tuple[Image.Image, dict]:
    """Read EXIF-oriented RGB image and calculate descriptive quality proxies.

    Flags are prompts for human review; they are NOT validated acceptance rules.
    A small source resolution can be perfectly usable, and a sharp image can be
    entirely outside the model's supported domain.
    """
    if not contents or len(contents) > MAX_FILE_BYTES:
        raise ValueError("Photo is empty or exceeds the 12 MB local prototype limit.")
    try:
        with Image.open(BytesIO(contents)) as source:
            w, h = source.size
            if w < 1 or h < 1 or w * h > MAX_IMAGE_PIXELS:
                raise ValueError("Image dimensions are invalid or exceed 20 megapixels.")
            photo = ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Cannot read this image safely; try a normal JPEG or PNG.") from exc
    w, h = photo.size
    thumbnail = photo.copy()
    thumbnail.thumbnail((256, 256))
    grayscale = np.asarray(thumbnail.convert("L"), dtype=np.float32)
    edges = np.asarray(thumbnail.convert("L").filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    mean_brightness = float(grayscale.mean())
    gray_std = float(grayscale.std())
    edge_mean = float(edges.mean())
    flags: list[str] = []
    if min(w, h) < 224:
        flags.append("At least one native dimension is below the model's 224 px input size.")
    if mean_brightness < 40:
        flags.append("Very dark image proxy — check visibility.")
    if mean_brightness > 215:
        flags.append("Very bright image proxy — check highlights.")
    if gray_std < 18:
        flags.append("Low contrast proxy — check visibility.")
    if edge_mean < 12:
        flags.append("Low edge-detail proxy — inspect focus and image content.")

    return photo, {
        "width_px": int(w),
        "height_px": int(h),
        "file_bytes": len(contents),
        "mean_brightness_0_255": round(mean_brightness, 3),
        "grayscale_std": round(gray_std, 3),
        "edge_mean": round(edge_mean, 3),
        "quality_flags": flags,
        "quality_assessment": "unvalidated_descriptive_flags_only",
    }


def make_review_record(
    *,
    image_hash: str,
    photo_metrics: dict,
    model_hash: str,
    probabilities: list[float],
    reference_label: str,
    reviewer_code: str,
    review_method: str,
    site_code: str,
    field_code: str,
    plant_code: str,
    session_code: str,
    device_code: str,
    lighting: str,
    consent_reference: str,
    notes: str = "",
) -> dict:
    if reference_label not in REFERENCE_LABELS:
        raise ValueError("Unsupported reference label")
    if len(probabilities) != len(MODEL_CLASSES) or not np.all(np.isfinite(probabilities)):
        raise ValueError("Expected three finite class scores")
    if any(p < 0 or p > 1 for p in probabilities):
        raise ValueError("Class scores must be within [0, 1]")
    if not np.isclose(sum(probabilities), 1.0, atol=1e-3):
        raise ValueError("Three class scores must sum to approximately 1")
    if not reviewer_code.strip() or not consent_reference.strip():
        raise ValueError("Reviewer code and permission reference are required")
    for field_value in (reviewer_code, review_method, site_code, field_code, plant_code,
                        session_code, device_code, lighting, consent_reference):
        if len(field_value) > 120:
            raise ValueError("One of the input code fields is too long")
    result = {
        "record_id": str(uuid.uuid4()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "app_version": VERSION,
        "study_mode": "shadow_expert_review_no_crop_advice",
        "image_sha256": image_hash,
        "native_image_metrics": photo_metrics,
        "reference_label": reference_label,
        "reference_status": ("unconfirmed" if reference_label == "Uncertain_unconfirmed" else "reviewer_entered"),
        "reviewer_code": reviewer_code.strip(),
        "review_method": review_method,
        "site_code": site_code.strip(),
        "field_code": field_code.strip(),
        "plant_code": plant_code.strip(),
        "session_code": session_code.strip(),
        "device_code": device_code.strip(),
        "lighting": lighting,
        "consent_reference": consent_reference.strip(),
        "notes": notes.strip()[:500],
        "model_sha256": model_hash,
        "input_pipeline": "EXIF-oriented RGB pixels 0..255, tf.image.resize bilinear 224x224; preprocessing in model",
        "class_order": list(MODEL_CLASSES),
        "model_scores_uncalibrated": {
            name: round(float(score), 8) for name, score in zip(MODEL_CLASSES, probabilities)
        },
        "highest_score_label": MODEL_CLASSES[int(np.argmax(probabilities))],
        "expert_review_required": True,
        "autonomous_field_diagnosis": False,
        "stored_original_image": False,
    }
    return result


def append_review_jsonl(path: Path, record: dict) -> None:
    """Append one locally entered record; NEVER write an uploaded image.

    Only use on one trusted local Streamlit server. This is not multi-user or
    network-safe storage and has no authentication, retention or encryption.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure JSON serialisability before opening output.
    line = json.dumps(record, ensure_ascii=False, allow_nan=False)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")
        handle.flush()
