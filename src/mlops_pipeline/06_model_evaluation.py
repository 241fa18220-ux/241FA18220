"""Step 6 - evaluate the best model on the held-out test set."""
import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from validation import ROOT, load_config, path, write_json  # noqa: E402

cfg = load_config()
proc = ROOT / cfg["paths"]["processed_dir"]
model = joblib.load(ROOT / cfg["paths"]["models_dir"] / "covid_model.pkl")
Xte, yte, ytr = np.load(proc / "X_test_final.npy"), np.load(proc / "y_test.npy"), np.load(proc / "y_train.npy")

pred = np.clip(model.predict(Xte), 0, None)
rmse = float(np.sqrt(mean_squared_error(yte, pred)))
baseline_rmse = float(np.sqrt(mean_squared_error(yte, np.full_like(yte, ytr.mean(), dtype=float))))
metrics = {"mae": float(mean_absolute_error(yte, pred)), "rmse": rmse, "r2": float(r2_score(yte, pred)),
           "baseline_rmse": baseline_rmse, "rmse_vs_baseline": rmse / baseline_rmse}
write_json("reports/evaluation_report.json", metrics)

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(yte, pred, alpha=.7)
lim = [0, max(yte.max(), pred.max()) * 1.05]
ax.plot(lim, lim, "--", c="gray"); ax.set_xlabel("Actual Recovered"); ax.set_ylabel("Predicted Recovered")
ax.set_title(f"Actual vs Predicted (R2 = {metrics['r2']:.3f})")
fig.tight_layout(); fig.savefig(path("artifacts/actual_vs_predicted.png"), dpi=120); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.5, 4))
ax.scatter(pred, yte - pred, alpha=.7); ax.axhline(0, c="gray", ls="--")
ax.set_xlabel("Predicted"); ax.set_ylabel("Residual")
fig.tight_layout(); fig.savefig(path("artifacts/residuals.png"), dpi=120); plt.close(fig)
print("[06]", {k: round(v, 3) for k, v in metrics.items()})
