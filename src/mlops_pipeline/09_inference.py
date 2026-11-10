"""Step 9 - batch inference on new country data."""
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from validation import ROOT, clean_dataframe, features, load_config, path, validate_schema

cfg = load_config()
m = ROOT / cfg["paths"]["models_dir"]
pre, model = joblib.load(m / "preprocessor.pkl"), joblib.load(m / "covid_model.pkl")
df = clean_dataframe(pd.read_csv(ROOT / cfg["paths"]["new_data"]), cfg)
issues = validate_schema(df, cfg, need_target=False)
assert not issues, issues

pred = np.clip(model.predict(pre.transform(df[features(cfg)])), 0, None).round().astype(int)
out = df[[cfg["id_column"], "Confirmed"]].assign(predicted_recovered=pred)
out.to_csv(path("artifacts/new_data_predictions.csv"), index=False)
with open(path("logs/inference/inference_log.jsonl"), "a") as f:
    f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "rows": len(out),
                        "mean_predicted_recovered": float(pred.mean())}) + "\n")
print(f"[09] predicted Recovered for {len(out)} countries | mean = {pred.mean():,.0f}")
