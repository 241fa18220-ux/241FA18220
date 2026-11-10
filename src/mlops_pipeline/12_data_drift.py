"""Data drift: PSI per numeric feature + category-share shift for WHO Region."""
import numpy as np
import pandas as pd
from validation import ROOT, clean_dataframe, load_config, path, write_json


def psi(base, new, bins=8):
    edges = np.unique(np.quantile(base.dropna(), np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    b = np.histogram(base.dropna(), edges)[0] / base.notna().sum()
    n = np.histogram(new.dropna(), edges)[0] / new.notna().sum()
    b, n = np.clip(b, 1e-6, None), np.clip(n, 1e-6, None)
    return float(np.sum((n - b) * np.log(n / b)))


cfg = load_config()
base = clean_dataframe(pd.read_csv(ROOT / cfg["paths"]["processed_dir"] / "X_train_raw.csv"), cfg)
new = clean_dataframe(pd.read_csv(ROOT / cfg["paths"]["drift_dir"] / "evolved_data.csv"), cfg)
rows = [{"feature": c, "type": "numeric", "score": psi(base[c], new[c])} for c in cfg["numeric_features"]]
for c in cfg["categorical_features"]:
    p, q = base[c].value_counts(normalize=True), new[c].value_counts(normalize=True)
    cats = p.index.union(q.index)
    rows.append({"feature": c, "type": "categorical",
                 "score": float(0.5 * np.abs(p.reindex(cats, fill_value=0) - q.reindex(cats, fill_value=0)).sum())})
res = pd.DataFrame(rows)
res["drifted"] = res["score"] > cfg["drift"]["psi_threshold"]
res.to_csv(path("artifacts/data_drift_report.csv"), index=False)
write_json("reports/data_drift_report.json", {"drifted_features": res[res.drifted].feature.tolist(),
                                              "any_drift": bool(res.drifted.any())})
print(res.round(3).to_string(index=False))
