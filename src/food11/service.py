"""FastAPI image-classification service for the Food-11 champion model."""

from __future__ import annotations

import io
import os
from html import escape
from contextlib import asynccontextmanager

import mlflow
import mlflow.pyfunc
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, ImageOps, UnidentifiedImageError


CLASSES = (
    "Bread",
    "Dairy product",
    "Dessert",
    "Egg",
    "Fried food",
    "Meat",
    "Noodles-Pasta",
    "Rice",
    "Seafood",
    "Soup",
    "Vegetable-Fruit",
)
IMAGE_SIZE = (128, 128)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_URI = os.getenv("MLFLOW_MODEL_URI", "models:/food11@champion")


def load_champion():
    mlflow.set_tracking_uri(TRACKING_URI)
    return mlflow.pyfunc.load_model(MODEL_URI)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_champion()
    if MODEL_URI.startswith("models:/") and "@" in MODEL_URI:
        model_name, alias = MODEL_URI.removeprefix("models:/").split("@", 1)
        app.state.model_version = mlflow.MlflowClient().get_model_version_by_alias(
            model_name, alias
        ).version
    else:
        app.state.model_version = "bundled"
    yield


app = FastAPI(
    title="Food-11 Champion Classifier",
    description="Upload a food photo and classify it with the MLflow champion model.",
    version="1.0.0",
    lifespan=lifespan,
)


def preprocess_image(raw: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(raw)) as source:
            image = ImageOps.fit(
                source.convert("RGB"), IMAGE_SIZE, method=Image.Resampling.LANCZOS
            )
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image") from exc

    array = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
    std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
    array = (array - mean) / std
    return np.transpose(array, (2, 0, 1))[None, ...]


def probabilities_from_prediction(prediction: object) -> np.ndarray:
    values = prediction.detach().cpu().numpy() if hasattr(prediction, "detach") else np.asarray(prediction)
    logits = np.asarray(values, dtype=np.float64).reshape(-1)
    if logits.size != len(CLASSES):
        raise RuntimeError(f"Expected {len(CLASSES)} model outputs, received {logits.size}")
    shifted = logits - logits.max()
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum()


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "model_uri": MODEL_URI,
        "model_version": app.state.model_version,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, object]:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload an image file")
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if not raw:
        raise HTTPException(status_code=400, detail="The uploaded file is empty")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MB limit")

    model_input = preprocess_image(raw)
    prediction = app.state.model.predict(model_input)
    probabilities = probabilities_from_prediction(prediction)
    order = np.argsort(probabilities)[::-1][:3]
    return {
        "predicted_category": CLASSES[int(order[0])],
        "confidence": round(float(probabilities[order[0]]), 6),
        "top_predictions": [
            {"category": CLASSES[int(index)], "confidence": round(float(probabilities[index]), 6)}
            for index in order
        ],
        "model_uri": MODEL_URI,
        "model_version": app.state.model_version,
    }


def render_upload_page(result: str = "Waiting for an image.") -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Food-11 Classifier</title>
  <style>
    :root {{ color-scheme: dark; font-family: Inter, system-ui, sans-serif; }}
    body {{ margin: 0; min-height: 100vh; display: grid; place-items: center;
      background: radial-gradient(circle at top, #19324a, #08111e 55%); color: #eef6ff; }}
    main {{ width: min(92vw, 620px); padding: 2rem; border: 1px solid #34516c;
      border-radius: 22px; background: rgba(11, 27, 43, .9); box-shadow: 0 24px 70px #0008; }}
    h1 {{ margin: 0 0 .5rem; }} p {{ color: #aac0d4; }}
    label {{ display: block; margin-bottom: .6rem; font-weight: 700; }}
    input {{ display: block; width: 100%; box-sizing: border-box; padding: 1rem;
      border: 2px solid #4f86b6; border-radius: 12px; background: #102940; color: #eef6ff; }}
    button {{ width: 100%; margin-top: 1rem; padding: .9rem;
      border: 0; border-radius: 12px; font-weight: 700; background: #42d392; color: #062014; cursor: pointer; }}
    #result {{ margin-top: 1rem; padding: 1rem; border-radius: 12px;
      background: #081725; white-space: pre-wrap; }}
  </style>
</head>
<body><main>
  <h1>Food-11 Champion</h1>
  <p>Upload a food image to classify it with the best MLflow model.</p>
  <form method="post" action="/" enctype="multipart/form-data">
    <label for="file">Choose a JPG or PNG food image</label>
    <input id="file" name="file" type="file" accept="image/jpeg,image/png,image/webp" required>
    <button type="submit">Upload and predict</button>
  </form>
  <div id="result">{result}</div>
</main></body></html>"""


@app.get("/", response_class=HTMLResponse)
def upload_page() -> str:
    return render_upload_page()


@app.post("/", response_class=HTMLResponse)
async def upload_and_predict(file: UploadFile = File(...)) -> str:
    prediction = await predict(file)
    top_predictions = prediction["top_predictions"]
    lines = "".join(
        f"<li>{escape(str(item['category']))}: {float(item['confidence']) * 100:.2f}%</li>"
        for item in top_predictions
    )
    result = (
        f"<strong>Prediction: {escape(str(prediction['predicted_category']))}</strong><br>"
        f"Confidence: {float(prediction['confidence']) * 100:.2f}%"
        f"<h3>Top 3</h3><ol>{lines}</ol>"
    )
    return render_upload_page(result)
