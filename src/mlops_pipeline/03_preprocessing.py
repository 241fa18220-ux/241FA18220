"""Step 3 - split (80/20, seed 42 like the notebook), build preprocessor, save arrays."""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from validation import ROOT, clean_dataframe, features, load_config, path, write_json

cfg = load_config()
num, cat, target, idc = cfg["numeric_features"], cfg["categorical_features"], cfg["target"], cfg["id_column"]
df = clean_dataframe(pd.read_csv(ROOT / cfg["paths"]["raw_data"]), cfg).drop_duplicates()

X, y = df[[idc] + features(cfg)], df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=cfg["test_size"], random_state=cfg["random_state"])


def build_preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
    ])


pre = build_preprocessor()
Xtr, Xte = pre.fit_transform(X_train[features(cfg)]), pre.transform(X_test[features(cfg)])

proc = cfg["paths"]["processed_dir"]
np.save(path(f"{proc}/X_train_final.npy"), Xtr)
np.save(path(f"{proc}/X_test_final.npy"), Xte)
np.save(path(f"{proc}/y_train.npy"), y_train.to_numpy())
np.save(path(f"{proc}/y_test.npy"), y_test.to_numpy())
X_train.assign(**{target: y_train}).to_csv(path(f"{proc}/X_train_raw.csv"), index=False)
X_test.assign(**{target: y_test}).to_csv(path(f"{proc}/X_test_raw.csv"), index=False)
X_test[[idc]].to_csv(path(f"{proc}/test_countries.csv"), index=False)
joblib.dump(pre, path(f"{cfg['paths']['models_dir']}/preprocessor.pkl"))

names = list(pre.get_feature_names_out())
write_json(f"{proc}/dataset_metadata.json", {
    "n_train": len(X_train), "n_test": len(X_test), "n_features": len(names), "feature_names": names,
    "target_mean_train": float(y_train.mean()), "target_mean_test": float(y_test.mean())})
write_json("artifacts/preprocessing_summary_report.json", {
    "train_shape": list(Xtr.shape), "test_shape": list(Xte.shape),
    "nan_in_train": int(np.isnan(Xtr).sum()), "status": "ok"})
print(f"[03] train {Xtr.shape} | test {Xte.shape}")
