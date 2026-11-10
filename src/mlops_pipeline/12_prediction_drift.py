"""Prediction drift: compare mean predicted Recovered on baseline vs evolved data."""
import joblib
import numpy as np
import pandas as pd
from validation import ROOT, clean_dataframe, features, load_config, write_json

cfg = load_config()
m = ROOT / cfg["paths"]["models_dir"]
pre, model = joblib.load(m / "preprocessor.pkl"), joblib.load(m / "covid_model.pkl")


def mean_pred(p):
    df = clean_dataframe(pd.read_csv(p), cfg)
    return float(np.clip(model.predict(pre.transform(df[features(cfg)])), 0, None).mean())


b = mean_pred(ROOT / cfg["paths"]["processed_dir"] / "X_test_raw.csv")
n = mean_pred(ROOT / cfg["paths"]["drift_dir"] / "evolved_data.csv")
write_json("reports/prediction_drift_report.json", {"baseline_mean_pred": b, "evolved_mean_pred": n, "ratio": n / b})
print(f"[12] mean predicted Recovered baseline={b:,.0f} evolved={n:,.0f} (x{n / b:.2f})")
