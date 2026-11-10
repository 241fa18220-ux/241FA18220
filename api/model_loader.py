import json
from functools import lru_cache
from pathlib import Path

import joblib

DEPLOY = Path(__file__).resolve().parents[1] / "deployment"


@lru_cache(maxsize=1)
def load_artifacts():
    model = joblib.load(DEPLOY / "model.pkl")
    pre = joblib.load(DEPLOY / "preprocessor.pkl")
    meta_path = DEPLOY / "model_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return model, pre, meta
