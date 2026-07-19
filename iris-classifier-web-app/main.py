"""
FastAPI backend for the Iris Classifier web application.
Loads a pre-trained PyTorch model and exposes a /predict endpoint.
"""

import os
from pathlib import Path

import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from model import IrisClassifier

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "iris_classification_model.pth"

CLASS_NAMES = {
    0: "Iris Setosa",
    1: "Iris Versicolor",
    2: "Iris Virginica",
}

CLASS_DESCRIPTIONS = {
    0: "Küçük, beyaz-mor çiçeklere sahip dayanıklı bir tür.",
    1: "Orta büyüklükte, mor-mavi çiçeklere sahip bir tür.",
    2: "En büyük çiçeklere sahip, derin mor tonlarında bir tür.",
}

CLASS_COLORS = {
    0: "#4ade80",   # green
    1: "#60a5fa",   # blue
    2: "#f472b6",   # pink/purple
}

# ── Model loading ──────────────────────────────────────────────────────────────
def load_model() -> IrisClassifier:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model dosyası bulunamadı: {MODEL_PATH}\n"
            "Lütfen 'iris_classification_model.pth' dosyasını bu dizine koyun."
        )
    m = IrisClassifier()
    m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    m.eval()
    return m


model = load_model()

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Iris Classifier API",
    description="PyTorch tabanlı Iris çiçeği sınıflandırıcısı",
    version="1.0.0",
)

# Serve static files (frontend)
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ── Schemas ────────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    sepal_length: float = Field(..., ge=0.0, le=20.0, description="Çanak yaprak uzunluğu (cm)")
    sepal_width: float  = Field(..., ge=0.0, le=20.0, description="Çanak yaprak genişliği (cm)")
    petal_length: float = Field(..., ge=0.0, le=20.0, description="Taç yaprak uzunluğu (cm)")
    petal_width: float  = Field(..., ge=0.0, le=20.0, description="Taç yaprak genişliği (cm)")


class PredictResponse(BaseModel):
    predicted_class: int
    class_name: str
    description: str
    color: str
    confidence: float
    probabilities: dict[str, float]


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(str(static_dir / "index.html"))


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    features = torch.tensor(
        [[req.sepal_length, req.sepal_width, req.petal_length, req.petal_width]],
        dtype=torch.float32,
    )
    with torch.inference_mode():
        logits = model(features)
        probs  = torch.softmax(logits, dim=1).squeeze()
        pred   = int(probs.argmax().item())

    prob_dict = {CLASS_NAMES[i]: round(float(probs[i].item()) * 100, 2) for i in range(3)}
    confidence = round(float(probs[pred].item()) * 100, 2)

    return PredictResponse(
        predicted_class=pred,
        class_name=CLASS_NAMES[pred],
        description=CLASS_DESCRIPTIONS[pred],
        color=CLASS_COLORS[pred],
        confidence=confidence,
        probabilities=prob_dict,
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model": "IrisClassifier", "classes": list(CLASS_NAMES.values())}
