"""Decide whether to retrain, based on drift + performance reports."""
from validation import load_config, read_json, write_json

cfg = load_config()
drift, perf = read_json("reports/data_drift_report.json"), read_json("reports/performance_report.json")
reasons = []
if drift["any_drift"]:
    reasons.append(f"data drift in {drift['drifted_features']}")
if perf["r2_drop"] > cfg["drift"]["max_r2_drop"]:
    reasons.append(f"R2 dropped by {perf['r2_drop']:.3f}")
write_json("reports/retrain_decision.json", {"retrain": bool(reasons), "reasons": reasons})
print("[12] RETRAIN" if reasons else "[12] model healthy", reasons)
