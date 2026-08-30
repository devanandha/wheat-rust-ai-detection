from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

from src.data import EXPECTED_CLASSES, IMAGE_SIZE

MODEL_PATH = Path("models/wheat_rust_mobilenetv2.keras")

st.set_page_config(page_title="Wheat Rust AI", page_icon="🌾", layout="centered")
st.title("🌾 Wheat Rust AI Detection")
st.caption("Research demonstration: Healthy, Brown Rust, or Yellow Rust")

with st.expander("Model and evaluation information"):
    st.write(
        "MobileNetV2 transfer-learning classifier trained on 2,942 images and "
        "evaluated on a separate 737-image validation split. Verified validation "
        "accuracy: 98.78% (728/737). This is not confirmed field accuracy."
    )


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


uploaded = st.file_uploader("Upload a wheat-leaf image", type=["jpg", "jpeg", "png"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", width="stretch")

    if not MODEL_PATH.exists():
        st.warning("The trained model is not available yet. Train it using the README instructions.")
    else:
        resized = image.resize(IMAGE_SIZE)
        batch = np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)
        probabilities = load_model().predict(batch, verbose=0)[0]
        index = int(np.argmax(probabilities))
        confidence = float(probabilities[index])
        st.subheader(EXPECTED_CLASSES[index].replace("_", " "))
        st.metric("Prediction confidence", f"{confidence:.1%}")
        probability_table = pd.DataFrame(
            {
                "Class": [name.replace("_", " ") for name in EXPECTED_CLASSES],
                "Probability": probabilities,
            }
        ).set_index("Class")
        st.bar_chart(probability_table)
        if confidence < 0.65:
            st.info("Low-confidence result: seek additional images or expert review.")
        st.caption("Confidence is a model score and is not the probability that a crop diagnosis is correct.")

st.divider()
st.caption("Educational research prototype only. It is not a substitute for professional crop assessment.")
