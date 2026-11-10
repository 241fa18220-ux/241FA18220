"""Performance on labeled evolved data."""
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from validation import ROOT, clean_dataframe, features, load_config, read_json, write_json

cfg = load_config()
m = ROOT / cfg["paths"]["models_dir"]
pre, model = joblib.load(m / "preprocessor.pkl"), joblib.load(m / "covid_model.pkl")
df = clean_dataframe(pd.read_csv(ROOT / cfg["paths"]["drift_dir"] / "evolved_labeled_data.csv"), cfg)
pred = np.clip(model.predict(pre.transform(df[features(cfg)])), 0, None)
y = df[cfg["target"]]
r2, rmse = r2_score(y, pred), float(np.sqrt(mean_squared_error(y, pred)))
base_r2 = read_json("reports/evaluation_report.json")["r2"]
write_json("reports/performance_report.json", {"baseline_r2": base_r2, "current_r2": r2,
                                               "current_rmse": rmse, "r2_drop": base_r2 - r2})
print(f"[12] R2 baseline={base_r2:.3f} current={r2:.3f}")
