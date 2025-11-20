from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import os
import joblib
import numpy as np
from langdetect import detect, DetectorFactory
import re

# Make langdetect deterministic
DetectorFactory.seed = 42

DEFAULT_BASELINE = os.path.join(os.path.dirname(__file__), "model", "baseline.joblib")
DEFAULT_OLD = os.path.join(os.path.dirname(__file__), "model", "model.joblib")
ENV_MODEL_PATH = os.environ.get("MODEL_PATH", "").strip()
ENV_MODEL_THRESHOLD = os.environ.get("MODEL_THRESHOLD", "").strip()
# Server-side default decision threshold (used if MODEL_THRESHOLD not set)
SERVER_DEFAULT_THRESHOLD = float(os.environ.get("SERVER_DEFAULT_THRESHOLD", "0.85"))
_pipeline = None
_label_names = None
_threshold = None
_pos_index = None

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")
_EMOJI_RE = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)

def _normalize_text(t: str) -> str:
    t = str(t)
    t = t.replace("\u200d", " ")
    t = _URL_RE.sub(" ", t)
    t = _EMOJI_RE.sub(" ", t)
    t = t.strip().lower()
    t = _WS_RE.sub(" ", t)
    return t

def load_model():
    global _pipeline, _label_names, _threshold, _pos_index
    candidates = []
    if ENV_MODEL_PATH:
        candidates.append(ENV_MODEL_PATH)
    candidates.append(DEFAULT_BASELINE)
    candidates.append(DEFAULT_OLD)
    for path in candidates:
        if path and os.path.exists(path):
            obj = joblib.load(path)
            _pipeline = obj.get("pipeline")
            _label_names = obj.get("label_names")
            _threshold = obj.get("threshold")
            _pos_index = obj.get("positive_index")
            break
    # Optional environment override for threshold
    if ENV_MODEL_THRESHOLD:
        try:
            _threshold = float(ENV_MODEL_THRESHOLD)
        except Exception:
            pass
    else:
        # If no explicit MODEL_THRESHOLD provided, use stricter server default
        _threshold = SERVER_DEFAULT_THRESHOLD

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    load_model()
    yield
    # Shutdown (if needed)

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictIn(BaseModel):
    text: str

class PredictOut(BaseModel):
    label: str
    confidence: float

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _pipeline is not None,
        "threshold": _threshold,
        "labels": _label_names,
    }

@app.post("/predict", response_model=PredictOut)
def predict(inp: PredictIn, th: float | None = None):
    if not inp.text or not inp.text.strip():
        raise HTTPException(status_code=400, detail="empty text")
    # Language guard: serve only Hindi/English
    try:
        lang = detect(inp.text)
    except Exception:
        lang = "unk"
    if lang not in {"hi", "en"}:
        raise HTTPException(status_code=400, detail=f"unsupported language: {lang}. Only Hindi/English are supported.")
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="model not loaded. run training.")
    probs = None
    text_norm = _normalize_text(inp.text)
    if hasattr(_pipeline, "predict_proba"):
        probs = _pipeline.predict_proba([text_norm])[0]
        # If we have a tuned threshold and pos_index for binary classification
        use_th = float(th) if th is not None else (float(_threshold) if _threshold is not None else 0.5)
        if len(probs) == 2:
            # Prefer anchoring on 'real' label prob if available, else fallback
            real_idx = None
            try:
                if _label_names and len(_label_names) == 2:
                    real_idx = int(list(_label_names).index("real"))
            except Exception:
                real_idx = None
            pos_idx = real_idx if real_idx is not None else (int(_pos_index) if _pos_index is not None else 1)
            pos_p = float(probs[pos_idx])
            idx = pos_idx if pos_p >= use_th else int(1 - pos_idx)
            conf = max(pos_p, 1.0 - pos_p)
        else:
            idx = int(np.argmax(probs))
            conf = float(probs[idx])
    else:
        scores = _pipeline.decision_function([text_norm])
        if scores.ndim == 1:
            s = float(scores[0])
            conf = float(1 / (1 + np.exp(-abs(s))))
            idx = int(1 if s >= 0 else 0)
        else:
            idx = int(np.argmax(scores[0]))
            m = float(np.max(scores[0]))
            conf = float(1 / (1 + np.exp(-m)))
            conf = 1 - conf
    label = _label_names[idx] if _label_names else str(idx)
    return {"label": label, "confidence": round(conf, 4)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
