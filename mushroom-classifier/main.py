"""
FastAPI backend for the Mushroom Classifier web app.

Loads the ANN exported from `mushroom-classification.ipynb` and answers
predictions for *partially* filled feature sets: the user may describe as many
or as few of the 20 features as they like.

How a partial answer is produced
--------------------------------
The model needs all 94 one-hot columns, so the unknown ones are marginalised out
instead of being guessed:

  • conditional  – every real mushroom in the dataset that matches what the user
                   entered is scored by the ANN and the scores are averaged.
                   P(poisonous) is then a genuine conditional probability, which
                   is why a half-filled form can legitimately answer "%51".
  • imputed      – if the described combination never occurs in the dataset, the
                   ANN is asked directly: the whole dataset is re-encoded with
                   the user's values forced into place and the outputs averaged.
  • blend        – few (but non-zero) matches: a support-weighted mix of the two.

Run:  uvicorn main:app --reload
"""

import json
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from model import MushroomANN

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
MODEL_PATH = HERE / "mushroom_model.pth"
META_PATH = HERE / "model_meta.json"
DATA_PATH = HERE / "encoded_dataset.npz"
INSIGHTS_PATH = HERE / "insights.json"
STATIC_DIR = HERE / "static"

MIN_SUPPORT = 12  # matching mushrooms needed before we trust the conditional
CERTAINTY_TIERS = [
    (99.5, "kesin", "Bu kadar bilgiyle sonuç neredeyse tartışmasız."),
    (90.0, "çok yüksek", "Girdiğin özellikler çok güçlü bir sinyal veriyor."),
    (75.0, "yüksek", "Sonuç oldukça net ama birkaç özellik daha ekleyebilirsin."),
    (60.0, "orta", "Eğilim belli ama kesin konuşmak için erken."),
    (0.0, "düşük", "Neredeyse yazı tura. Birkaç özellik daha gir."),
]

# ── Artefacts ─────────────────────────────────────────────────────────────────
for path in (MODEL_PATH, META_PATH, DATA_PATH, INSIGHTS_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} bulunamadı. Önce `python export_model.py` çalıştır."
        )

META = json.loads(META_PATH.read_text())
INSIGHTS = json.loads(INSIGHTS_PATH.read_text())

FEATURE_COLUMNS: list[str] = META["feature_columns"]
COL_INDEX = {name: i for i, name in enumerate(FEATURE_COLUMNS)}
FEATURES = {f["key"]: f for f in META["features"]}
FEATURE_ORDER = [f["key"] for f in META["features"]]
IMPORTANCE = {f["key"]: f["cramers_v"] for f in INSIGHTS["feature_importance"]}

_data = np.load(DATA_PATH, allow_pickle=False)
X_ALL = _data["X"].astype(np.float32)          # (8124, 94) one-hot matrix
Y_ALL = _data["y"].astype(np.int64)            # (8124,) 0 = edible, 1 = poisonous
RAW_ALL = _data["raw"]                         # (8124, 20) original letter codes
RAW_COL = {key: i for i, key in enumerate(FEATURE_ORDER)}

model = MushroomANN(input_features=len(FEATURE_COLUMNS))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
model.eval()

# Every dataset row is scored once at start-up: conditional queries are then
# just a masked mean over these probabilities.
with torch.inference_mode():
    ROW_PROBS = model(torch.from_numpy(X_ALL)).squeeze(1).numpy().astype(np.float64)


# ── Inference helpers ─────────────────────────────────────────────────────────
def _match_mask(selection: dict[str, str]) -> np.ndarray:
    mask = np.ones(len(RAW_ALL), dtype=bool)
    for key, value in selection.items():
        mask &= RAW_ALL[:, RAW_COL[key]] == value
    return mask


def _imputed_prob(selection: dict[str, str]) -> float:
    """Force the user's values onto every row, let the ANN score the result."""
    X = X_ALL.copy()
    for key, value in selection.items():
        feature = FEATURES[key]
        for col in feature["columns"]:
            X[:, COL_INDEX[col]] = 0.0
        for col in next(v["columns"] for v in feature["values"] if v["code"] == value):
            X[:, COL_INDEX[col]] = 1.0
    with torch.inference_mode():
        return float(model(torch.from_numpy(X)).mean().item())


def _probability(selection: dict[str, str]) -> tuple[float, int, str]:
    """→ (P(poisonous) in 0..1, support, mode)"""
    if not selection:
        return float(ROW_PROBS.mean()), len(RAW_ALL), "prior"

    mask = _match_mask(selection)
    support = int(mask.sum())

    if support >= MIN_SUPPORT:
        return float(ROW_PROBS[mask].mean()), support, "conditional"

    imputed = _imputed_prob(selection)
    if support == 0:
        return imputed, 0, "imputed"

    w = support / MIN_SUPPORT
    conditional = float(ROW_PROBS[mask].mean())
    return w * conditional + (1 - w) * imputed, support, "blend"


def _entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def _contributions(selection: dict[str, str], p_full: float) -> list[dict]:
    """Leave-one-out: how much did each entered feature move the verdict?"""
    out = []
    for key, value in selection.items():
        reduced = {k: v for k, v in selection.items() if k != key}
        p_without, _, _ = _probability(reduced)
        feature = FEATURES[key]
        label = next(v["label"] for v in feature["values"] if v["code"] == value)
        out.append({
            "key": key,
            "label": feature["label"],
            "icon": feature["icon"],
            "value": value,
            "value_label": label,
            "delta": round((p_full - p_without) * 100, 1),
        })
    out.sort(key=lambda d: -abs(d["delta"]))
    return out


def _next_questions(selection: dict[str, str], p_full: float) -> list[dict]:
    """Which unanswered feature would cut the remaining uncertainty the most?"""
    remaining = [k for k in FEATURE_ORDER if k not in selection]
    if not remaining:
        return []

    mask = _match_mask(selection) if selection else np.ones(len(RAW_ALL), dtype=bool)
    support = int(mask.sum())
    h_now = _entropy(p_full)

    if support >= MIN_SUPPORT:
        if h_now <= 0.005:
            return []  # the verdict is already decisive, nothing left to ask
        probs = ROW_PROBS[mask]
        raw = RAW_ALL[mask]
        scored = []
        for key in remaining:
            col = raw[:, RAW_COL[key]]
            expected = 0.0
            for value in np.unique(col):
                sub = col == value
                w = sub.sum() / support
                expected += w * _entropy(float(probs[sub].mean()))
            scored.append((key, h_now - expected))
        scored = [(k, g) for k, g in scored if g > 0.005]
        scored.sort(key=lambda t: -t[1])
    else:  # too few matches to measure information gain → global importance
        scored = [(k, IMPORTANCE.get(k, 0.0)) for k in remaining]
        scored.sort(key=lambda t: -t[1])

    return [{
        "key": k,
        "label": FEATURES[k]["label"],
        "icon": FEATURES[k]["icon"],
        "gain": round(float(g), 3),
    } for k, g in scored[:3]]


def _certainty(confidence: float) -> tuple[str, str]:
    for threshold, name, note in CERTAINTY_TIERS:
        if confidence >= threshold:
            return name, note
    return CERTAINTY_TIERS[-1][1], CERTAINTY_TIERS[-1][2]


# ── API ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Mushroom Classifier API",
    description="PyTorch ANN ile mantar zehirli mi, yenilebilir mi?",
    version="1.0.0",
)

STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class PredictRequest(BaseModel):
    selection: dict[str, str] = Field(
        default_factory=dict,
        description="Kullanıcının girdiği özellikler: {'odor': 'f', 'habitat': 'd'}",
    )


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/meta")
async def meta():
    """Feature list + value options for building the form."""
    return {
        "features": META["features"],
        "input_features": META["input_features"],
        "n_params": META["n_params"],
        "model": INSIGHTS["model"],
        "dataset": INSIGHTS["dataset"],
    }


@app.get("/api/insights")
async def insights():
    return INSIGHTS


@app.post("/api/predict")
async def predict(req: PredictRequest):
    selection = {}
    for key, value in req.selection.items():
        if value in (None, "", "any"):
            continue
        if key not in FEATURES:
            raise HTTPException(400, f"Bilinmeyen özellik: {key}")
        if value not in [v["code"] for v in FEATURES[key]["values"]]:
            raise HTTPException(400, f"'{key}' için geçersiz değer: {value}")
        selection[key] = value

    p_poison, support, mode = _probability(selection)
    poison_pct = p_poison * 100
    verdict = "poisonous" if poison_pct >= 50 else "edible"
    confidence = poison_pct if verdict == "poisonous" else 100 - poison_pct
    tier, note = _certainty(confidence)

    return {
        "verdict": verdict,
        "poison_pct": round(poison_pct, 1),
        "edible_pct": round(100 - poison_pct, 1),
        "confidence": round(confidence, 1),
        "certainty": tier,
        "certainty_note": note,
        "entered": len(selection),
        "total_features": len(FEATURE_ORDER),
        "support": support,
        "mode": mode,
        "contributions": _contributions(selection, p_poison) if selection else [],
        "next_questions": _next_questions(selection, p_poison),
    }


@app.get("/api/random")
async def random_mushroom(n: int = 5):
    """A real mushroom from the dataset, with n of its features revealed."""
    i = int(np.random.randint(len(RAW_ALL)))
    row = {key: str(RAW_ALL[i, RAW_COL[key]]) for key in FEATURE_ORDER}
    n = max(1, min(n, len(FEATURE_ORDER)))
    keys = list(np.random.choice(FEATURE_ORDER, size=n, replace=False))
    return {
        "selection": {str(k): row[str(k)] for k in keys},
        "truth": "poisonous" if int(Y_ALL[i]) == 1 else "edible",
        "revealed": n,
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "MushroomANN",
        "input_features": len(FEATURE_COLUMNS),
        "params": META["n_params"],
        "rows": len(RAW_ALL),
    }
