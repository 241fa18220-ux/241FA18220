"""List MLflow experiments/runs (requires `pip install mlflow` and a prior training run)."""
import mlflow
from validation import ROOT

mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
for r in mlflow.search_runs(experiment_names=["covid-recovery-prediction"]).itertuples():
    print(r.run_id[:8], getattr(r, "params_model", None), getattr(r, "metrics_cv_rmse", None))
