"""Wheat Rust AI — local expert-in-the-loop, shadow-mode field prototype.

Run only on a trusted local computer. No login, remote server protection,
participant-consent workflow, or protected database has been implemented.
Do NOT deploy as a publicly accessible app or offer agronomic diagnosis.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf

from src.field_research import (
    MODEL_CLASSES,
    REFERENCE_LABELS,
    append_review_jsonl,
    inspect_photo,
    make_review_record,
    sha256_bytes,
    sha256_file,
)

LOG_PATH = Path("outputs/field_research/reviews.jsonl")
DEFAULT_MODEL_PATH = os.environ.get(
    "WHEAT_FIELD_MODEL", "models/wheat_rust_mobilenetv2.keras"
)

st.set_page_config(
    page_title="Wheat Rust AI — Field Research",
    page_icon="🌾",
    layout="wide",
)

st.title("🌾 Wheat Rust AI — Field Research")
st.caption("Local research-only prototype · Expert-first labels · Shadow-mode inference")
st.error(
    "**NOT FIELD-VALIDATED — DO NOT USE FOR CROP DIAGNOSIS OR TREATMENT.** "
    "The saved classifier performed poorly on a previously inspected external "
    "dataset (188/489 correct; Yellow Rust 15/239). Every sample requires "
    "independent expert review. No autonomous result should guide farm decisions."
)
st.info(
    "This app is intended for a trusted local researcher. It has no login, "
    "encrypted study database, or built-in consent-management system. Uploaded "
    "photo bytes are not saved; optional study metadata and AI scores are "
    "appended only on explicit submission."
)


@st.cache_resource(show_spinner="Loading the saved MobileNetV2 classifier…")
def load_saved_model(model_path: str):
    return tf.keras.models.load_model(model_path, compile=False)


def predict_for_research(model, photo) -> list[float]:
    # Match src.data.load_split: float RGB 0..255, bilinear to 224x224.
    # The existing Keras model includes MobileNetV2 preprocessing internally.
    pixels = tf.convert_to_tensor(np.asarray(photo, dtype=np.float32))
    resized = tf.image.resize(pixels, (224, 224), method="bilinear")
    scores = np.asarray(model(tf.expand_dims(resized, 0), training=False), dtype=float)
    if scores.shape != (1, 3) or not np.all(np.isfinite(scores)):
        raise ValueError(f"Unexpected model output shape or nonfinite scores: {scores.shape}")
    return list(map(float, scores[0]))


with st.sidebar:
    st.subheader("Study settings")
    model_path = st.text_input(
        "Saved Keras model path",
        value=DEFAULT_MODEL_PATH,
        help="Use the frozen research checkpoint, not a newly trained model.",
    )
    st.caption(
        "Only the provided model's three classes are supported: "
        "Brown Rust, Healthy and Yellow Rust. Other diseases or mixed "
        "infections are outside its label vocabulary."
    )
    st.divider()
    st.subheader("Local study log")
    st.code(str(LOG_PATH))
    st.caption(
        "A JSON Lines log containing image fingerprints, site codes, "
        "expert labels and model scores — **not the uploaded photo**. "
        "Store and share it carefully; hashes and field metadata may be sensitive."
    )
    if LOG_PATH.is_file():
        st.download_button(
            "Export local JSONL",
            LOG_PATH.read_bytes(),
            file_name="field_review_records.jsonl",
            mime="application/x-ndjson",
        )
    st.warning(
        "Run on your own laptop only (127.0.0.1). "
        "Do not deploy this unauthenticated prototype publicly."
    )

st.subheader("1 · Capture an image")
capture_mode = st.radio(
    "Source",
    ["Upload image", "Take photo using camera"],
    horizontal=True,
)
if capture_mode == "Take photo using camera":
    uploaded = st.camera_input("Capture a wheat-leaf image for review")
else:
    uploaded = st.file_uploader(
        "Select a wheat-leaf image", type=["jpg", "jpeg", "png"]
    )

if uploaded is None:
    st.caption("Select or capture a photo to begin a local, non-diagnostic study record.")
    st.stop()

raw_image = uploaded.getvalue()
try:
    image, metrics = inspect_photo(raw_image)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

image_hash = sha256_bytes(raw_image)

left, right = st.columns([1, 1])
with left:
    st.image(image, caption="Image under independent expert review", use_container_width=True)
with right:
    st.subheader("2 · Descriptive image checks")
    st.metric("Original dimensions", f"{metrics['width_px']} × {metrics['height_px']} px")
    st.write(f"Mean grayscale brightness: {metrics['mean_brightness_0_255']:.1f} / 255")
    st.write(f"Grayscale variation: {metrics['grayscale_std']:.1f}")
    st.write(f"Edge-detail proxy: {metrics['edge_mean']:.1f}")
    if metrics["quality_flags"]:
        for flag in metrics["quality_flags"]:
            st.warning(flag)
    else:
        st.info("No heuristic quality flags. This is NOT evidence the leaf or image is suitable.")
    st.caption(
        "Brightness/edge checks are unvalidated heuristics, not a blur detector, "
        "wheat-leaf detector or evidence of diagnostic reliability. "
        "Even apparently clear photographs can contain unsupported diseases."
    )

st.divider()
st.subheader("3 · Independent expert reference before seeing AI results")
st.caption(
    "Enter the reference label based on expert observation **before** revealing "
    "the model's prediction to reduce anchoring. Use 'Other or mixed' for unsupported "
    "presentations and 'Uncertain' when reliable ground truth is unavailable."
)

with st.form("expert_review_form", clear_on_submit=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        reviewer_code = st.text_input("Expert reviewer code *", placeholder="e.g. EXP-01")
        site_code = st.text_input("Study site code", placeholder="e.g. SITE-01")
        field_code = st.text_input("Field code", placeholder="e.g. FIELD-01")
        plant_code = st.text_input("Plant code", placeholder="e.g. PLANT-01")
    with col2:
        session_code = st.text_input("Capture session code", placeholder="e.g. SESSION-01")
        device_code = st.text_input("Phone / camera code", placeholder="e.g. DEVICE-01")
        lighting = st.selectbox(
            "Lighting", ["Not recorded", "Daylight", "Shade", "Artificial", "Mixed"]
        )
        review_method = st.selectbox(
            "Expert assessment method", ["Visual only", "Microscopy", "Other", "Not confirmed"]
        )
    with col3:
        reference_label = st.selectbox(
            "Expert reference label *",
            options=REFERENCE_LABELS,
            format_func=lambda x: {
                "Brown_rust": "Brown Rust",
                "Healthy": "Healthy",
                "Yellow_rust": "Yellow Rust",
                "Other_or_mixed": "Other disease or mixed symptoms",
                "Uncertain_unconfirmed": "Uncertain / not confirmed",
            }[x],
        )
        consent_reference = st.text_input(
            "Permission / consent reference *",
            help="Enter the study's permission reference, not a participant's name or address.",
        )
        notes = st.text_area(
            "Reviewer notes (optional; no personal details)", max_chars=500
        )
    permission_check = st.checkbox(
        "I have permission to process the image and save its local study metadata and AI output."
    )
    independent_check = st.checkbox(
        "I recorded the reference label independently, without seeing the AI output."
    )
    submitted = st.form_submit_button(
        "Save expert reference and run research analysis",
        type="primary",
    )

if submitted:
    if not reviewer_code.strip() or not consent_reference.strip():
        st.error("Reviewer code and permission reference are required.")
    elif not (permission_check and independent_check):
        st.error("Confirm study permission and independent labeling before proceeding.")
    elif not Path(model_path).is_file():
        st.error(f"Model not found at: {model_path}")
    else:
        try:
            with st.spinner("Running frozen-model inference locally…"):
                model = load_saved_model(str(Path(model_path).resolve()))
                probabilities = predict_for_research(model, image)
                record = make_review_record(
                    image_hash=image_hash,
                    photo_metrics=metrics,
                    model_hash=sha256_file(Path(model_path)),
                    probabilities=probabilities,
                    reference_label=reference_label,
                    reviewer_code=reviewer_code,
                    review_method=review_method,
                    site_code=site_code,
                    field_code=field_code,
                    plant_code=plant_code,
                    session_code=session_code,
                    device_code=device_code,
                    lighting=lighting,
                    consent_reference=consent_reference,
                    notes=notes,
                )
                append_review_jsonl(LOG_PATH, record)
            st.session_state["latest_field_record"] = record
            st.success("Expert-reviewed study record saved locally. Source image not saved.")
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            st.error(f"Unable to complete the research analysis: {exc}")

record = st.session_state.get("latest_field_record")
if record is not None and record["image_sha256"] == image_hash:
    st.divider()
    st.subheader("4 · Research-only prediction and disagreement")
    st.warning(
        "**EXPERT REVIEW REQUIRED.** This is an uncalibrated closed-set model result, "
        "not a crop diagnosis. Do not use it to determine treatment or reassure a "
        "farmer that a plant is healthy."
    )
    result_left, result_right = st.columns(2)
    with result_left:
        st.write("**Expert-entered reference:**", record["reference_label"].replace("_", " "))
        st.write("**Model highest-scoring class:**", record["highest_score_label"].replace("_", " "))
        if record["reference_label"] in MODEL_CLASSES:
            if record["reference_label"] != record["highest_score_label"]:
                st.error("Model and reviewer labels disagree — investigate this case.")
            else:
                st.info("Agreement with one reviewer is not a validated diagnosis.")
        else:
            st.info("The expert's label is outside the supported three-class model scope.")
    with result_right:
        st.write("**Uncalibrated model scores**")
        for name, score in record["model_scores_uncalibrated"].items():
            st.write(f"{name.replace('_', ' ')}: {score:.3f}")
    st.caption(
        "Photo hashes are used locally to identify repeat submissions. "
        "These and field codes are not automatically anonymous. "
        "Expert labels must be adjudicated and split by field/session "
        "before any scientific performance analysis."
    )
