"""Human-facing upload page for the Food-11 inference service."""

from __future__ import annotations

import os

import requests
import streamlit as st


INFERENCE_URL = os.getenv("INFERENCE_URL", "http://127.0.0.1:8000").rstrip("/")

st.title("Food-11 classifier")
st.caption("Upload a food photo and ask the champion model to classify it.")

uploaded = st.file_uploader("Food image", type=["jpg", "jpeg", "png", "webp"])

if uploaded is not None:
    st.image(uploaded, width=300)
    if st.button("Predict category", type="primary"):
        try:
            response = requests.post(
                f"{INFERENCE_URL}/predict",
                files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as exc:
            st.error(f"Inference service is unavailable: {exc}")
        else:
            st.success(
                f"Prediction: {result['predicted_category']} "
                f"({result['confidence']:.1%} confidence)"
            )
            st.caption(f"MLflow model version: {result['model_version']}")
            st.write("Top three categories")
            for item in result["top_predictions"]:
                st.write(f"{item['category']}: {item['confidence']:.1%}")
