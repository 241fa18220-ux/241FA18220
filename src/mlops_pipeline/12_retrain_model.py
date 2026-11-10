"""Retrain on original train data + new labeled data; keep v2 only if it is better."""
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from validation import ROOT, clean_dataframe, features, load_config, path, write_json

cfg = load_config()
target, md = cfg["target"], cfg["paths"]["models_dir"]
train = clean_dataframe(pd.read_csv(ROOT / cfg["paths"]["processed_dir"] / "X_train_raw.csv"), cfg)
new = clean_dataframe(pd.read_csv(ROOT / cfg["paths"]["drift_dir"] / "evolved_labeled_data.csv"), cfg)
combined = pd.concat([train, new], ignore_index=True)
combined.to_csv(path(f"{cfg['paths']['drift_dir']}/post_drift_training_data.csv"), index=False)
combined.to_csv(path(f"{cfg['paths']['processed_dir']}/updated_training_data.csv"), index=False)

tr, ho = train_test_split(combined, test_size=0.2, random_state=cfg["random_state"])
old_pre, old_model = joblib.load(ROOT / md / "preprocessor.pkl"), joblib.load(ROOT / md / "covid_model.pkl")
new_pre = clone(old_pre).fit(tr[features(cfg)])
new_model = clone(old_model).fit(new_pre.transform(tr[features(cfg)]), tr[target])


def score(pre, model):
    p = np.clip(model.predict(pre.transform(ho[features(cfg)])), 0, None)
    return float(r2_score(ho[target], p)), float(np.sqrt(mean_squared_error(ho[target], p)))


(old_r2, old_rmse), (new_r2, new_rmse) = score(old_pre, old_model), score(new_pre, new_model)
joblib.dump(new_model, path(f"{md}/retrained_model_v2.pkl"))
joblib.dump(new_pre, path(f"{md}/retrained_preprocessor_v2.pkl"))
write_json("reports/model_comparison.json", {"old_r2": old_r2, "new_r2": new_r2, "old_rmse": old_rmse,
                                             "new_rmse": new_rmse, "promote_v2": new_rmse < old_rmse})
print(f"[12] old R2={old_r2:.3f} RMSE={old_rmse:,.0f} | v2 R2={new_r2:.3f} RMSE={new_rmse:,.0f} | promote={new_rmse < old_rmse}")
