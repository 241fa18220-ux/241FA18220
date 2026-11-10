import time

import numpy as np
from fastapi import FastAPI, HTTPException

from .model_loader import load_artifacts
from .monitoring_logger import log_request
from .preprocessing import to_frame
from .schemas import BatchRequest, BatchResponse, Country, Prediction

app = FastAPI(title="COVID-19 Recovered Cases Prediction API", version="1.0.0")


def _predict(countries):
    model, pre, _ = load_artifacts()
    pred = np.clip(model.predict(pre.transform(to_frame(countries))), 0, None).round().astype(int)
    out = [Prediction(country=c.country, predicted_recovered=int(p),
                      predicted_recovery_pct=round(100 * int(p) / c.confirmed, 2) if c.confirmed else None)
           for c, p in zip(countries, pred)]
    return out, float(pred.mean())


@app.get("/health")
def health():
    try:
        _, _, meta = load_artifacts()
        return {"status": "ok", "deployed_at": meta.get("deployed_at"),
                "model": meta.get("model_report", {}).get("best_model")}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"model not loaded: {e}")


@app.post("/predict", response_model=Prediction)
def predict(country: Country):
    t = time.time()
    try:
        preds, mean_p = _predict([country])
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="model not deployed")
    log_request("/predict", 1, mean_p, t)
    return preds[0]


@app.post("/predict_batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    if not req.countries:
        raise HTTPException(status_code=422, detail="countries list is empty")
    t = time.time()
    preds, mean_p = _predict(req.countries)
    log_request("/predict_batch", len(preds), mean_p, t)
    return BatchResponse(predictions=preds)
