from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

from src.data import EXPECTED_CLASSES, IMAGE_SIZE
from src.gradcam import make_gradcam_heatmap, overlay_heatmap, heatmap_to_image

MODEL_PATH = Path("models/wheat_rust_mobilenetv2.keras")

st.set_page_config(
    page_title="Wheat Rust AI",
    page_icon="🌾",
    layout="centered",
)

st.title("🌾 Wheat Rust AI Detection")
st.caption(
    "Research demonstration: Healthy, Brown Rust, or Yellow Rust "
    "with Grad-CAM explainability"
)

with st.expander("Model and evaluation information"):
    st.write(
        "MobileNetV2 transfer-learning classifier trained on 2,942 images and "
        "evaluated on a separate 737-image validation split. Verified validation "
        "accuracy: 98.78% (728/737). This is not confirmed field accuracy."
    )
    st.write(
        "Grad-CAM is used to visualise image regions that influence the model's "
        "prediction. These visualisations are interpretability aids and should not "
        "be treated as proof of lesion localisation or biological causation."
    )


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


uploaded = st.file_uploader(
    "Upload a wheat-leaf image",
    type=["jpg", "jpeg", "png"],
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    if not MODEL_PATH.exists():
        st.warning(
            "The trained model is not available yet. "
            "Train it using the README instructions."
        )

    else:
        model = load_model()

        resized = image.resize(IMAGE_SIZE)
        batch = np.expand_dims(
            np.asarray(resized, dtype=np.float32),
            axis=0,
        )

        probabilities = model.predict(batch, verbose=0)[0]

        index = int(np.argmax(probabilities))
        confidence = float(probabilities[index])

        predicted_class = EXPECTED_CLASSES[index].replace("_", " ")

        st.subheader(f"Prediction: {predicted_class}")

        st.metric(
            "Prediction confidence",
            f"{confidence:.1%}",
        )

        probability_table = pd.DataFrame(
            {
                "Class": [
                    name.replace("_", " ")
                    for name in EXPECTED_CLASSES
                ],
                "Probability": probabilities,
            }
        ).set_index("Class")

        st.bar_chart(probability_table)

        if confidence < 0.65:
            st.info(
                "Low-confidence result: seek additional images "
                "or expert review."
            )

        st.caption(
            "Confidence is a model score and is not the probability "
            "that a crop diagnosis is correct."
        )

        st.divider()

        st.subheader("Model explanation")

        try:
            heatmap = make_gradcam_heatmap(
                model,
                batch,
                class_index=index,
            )

            heatmap_image = heatmap_to_image(
                heatmap,
                image.size,
            )

            overlay = overlay_heatmap(
                image,
                heatmap,
                alpha=0.40,
            )

            col1, col2 = st.columns(2)

            with col1:
                st.image(
                    image,
                    caption="Uploaded image",
                    width="stretch",
                )

            with col2:
                st.image(
                    overlay,
                    caption="Grad-CAM overlay",
                    width="stretch",
                )

            with st.expander("View attention heatmap"):
                st.image(
                    heatmap_image,
                    caption="Grad-CAM heatmap",
                    width="stretch",
                )

            st.caption(
                "Grad-CAM highlights image regions that influenced "
                "the selected prediction. It does not prove that these "
                "regions correspond to biologically confirmed disease lesions."
            )

        except Exception as exc:
            st.warning(
                "The prediction was generated successfully, "
                "but the Grad-CAM explanation could not be created."
            )
            st.code(str(exc))

st.divider()

st.caption(
    "Educational research prototype only. "
    "It is not a substitute for professional crop assessment."
)