"""Step 5 - compare the notebook's 4 regressors by 5-fold CV RMSE (train only), save the best."""
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from validation import ROOT, features, load_config, path, write_json

cfg = load_config()
rs, models_dir = cfg["random_state"], cfg["paths"]["models_dir"]
proc = ROOT / cfg["paths"]["processed_dir"]
Xtr, ytr = np.load(proc / "X_train_final.npy"), np.load(proc / "y_train.npy")
raw_train = pd.read_csv(proc / "X_train_raw.csv")
pre = joblib.load(ROOT / models_dir / "preprocessor.pkl")

candidates = {
    "linear_regression": LinearRegression(),
    "decision_tree": DecisionTreeRegressor(max_depth=5, random_state=rs),
    "random_forest": RandomForestRegressor(n_estimators=200, max_depth=5, random_state=rs),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=2, random_state=rs),
}

try:
    import mlflow
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
except Exception:  # MLflow optional
    mlflow = None

cv = KFold(5, shuffle=True, random_state=rs)
cv_rmse = {}
for name, model in candidates.items():
    pipe = Pipeline([("pre", clone(pre)), ("model", clone(model))])
    scores = cross_val_score(pipe, raw_train[features(cfg)], raw_train[cfg["target"]],
                             scoring="neg_root_mean_squared_error", cv=cv)
    cv_rmse[name] = float(-scores.mean())
    print(f"[05] {name:20s} CV RMSE = {cv_rmse[name]:,.0f}")
    if mlflow:
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model", name)
            mlflow.log_metric("cv_rmse", cv_rmse[name])

best_name = min(cv_rmse, key=cv_rmse.get)
best = clone(candidates[best_name]).fit(Xtr, ytr)
joblib.dump(best, path(f"{models_dir}/covid_model.pkl"))
for n in ("random_forest", "decision_tree"):
    joblib.dump(clone(candidates[n]).fit(Xtr, ytr), path(f"{models_dir}/{n}_baseline.pkl"))
joblib.dump(Pipeline([("pre", pre), ("model", best)]), path(f"{models_dir}/covid_full_pipeline.pkl"))
write_json("artifacts/production_model_report.json", {"cv_rmse": cv_rmse, "best_model": best_name})
print(f"[05] best = {best_name}")
