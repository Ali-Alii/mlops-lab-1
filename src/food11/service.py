"""FastAPI image-classification service for the Food-11 champion model."""

from __future__ import annotations

import io
import os
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
    return {"status": "ok", "model_uri": MODEL_URI}


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
    }


@app.get("/", response_class=HTMLResponse)
def upload_page() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Food-11 Classifier</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, system-ui, sans-serif; }
    body { margin: 0; min-height: 100vh; display: grid; place-items: center;
      background: radial-gradient(circle at top, #19324a, #08111e 55%); color: #eef6ff; }
    main { width: min(92vw, 620px); padding: 2rem; border: 1px solid #34516c;
      border-radius: 22px; background: rgba(11, 27, 43, .9); box-shadow: 0 24px 70px #0008; }
    h1 { margin: 0 0 .5rem; } p { color: #aac0d4; }
    label { display: block; padding: 2rem; border: 2px dashed #4f86b6; border-radius: 16px;
      text-align: center; cursor: pointer; background: #102940; }
    input { display: none; } button { width: 100%; margin-top: 1rem; padding: .9rem;
      border: 0; border-radius: 12px; font-weight: 700; background: #42d392; color: #062014; cursor: pointer; }
    button:disabled { opacity: .45; cursor: wait; } img { display: none; max-width: 100%; max-height: 280px;
      margin: 1rem auto; border-radius: 14px; } #result { margin-top: 1rem; padding: 1rem;
      border-radius: 12px; background: #081725; white-space: pre-wrap; }
  </style>
</head>
<body><main>
  <h1>Food-11 Champion</h1>
  <p>Upload a food image to classify it with the best MLflow model.</p>
  <form id="form" method="post" action="/predict" enctype="multipart/form-data">
    <label for="file">Choose or drop a JPG/PNG image<input id="file" name="file" type="file" accept="image/*" required></label>
    <img id="preview" alt="Selected food image">
    <button id="submit" type="submit">Predict category</button>
  </form>
  <div id="result">Waiting for an image.</div>
<script>
const form=document.querySelector('#form'), file=document.querySelector('#file'), preview=document.querySelector('#preview');
file.onchange=()=>{ if(file.files[0]) { preview.src=URL.createObjectURL(file.files[0]); preview.style.display='block'; } };
form.onsubmit=async(e)=>{ e.preventDefault(); const button=document.querySelector('#submit'), result=document.querySelector('#result');
  button.disabled=true; result.textContent='Running prediction...';
  try { const response=await fetch('/predict',{method:'POST',body:new FormData(form)}); const data=await response.json();
    if(!response.ok) throw new Error(data.detail || 'Prediction failed');
    result.textContent=`Prediction: ${data.predicted_category}\nConfidence: ${(data.confidence*100).toFixed(2)}%\n\nTop 3:\n`+
      data.top_predictions.map(x=>`${x.category}: ${(x.confidence*100).toFixed(2)}%`).join('\n');
  } catch(error) { result.textContent=`Error: ${error.message}`; } finally { button.disabled=false; }
};
</script></main></body></html>"""
