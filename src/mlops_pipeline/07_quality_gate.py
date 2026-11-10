"""Step 7 - quality gate: block promotion if the model is not good enough."""
import sys

from validation import load_config, read_json, write_json

cfg = load_config()
g, m = cfg["quality_gate"], read_json("reports/evaluation_report.json")
checks = {"r2": m["r2"] >= g["min_r2"], "rmse_vs_baseline": m["rmse_vs_baseline"] <= g["max_rmse_vs_baseline"]}
passed = all(checks.values())
write_json("reports/quality_gate_report.json", {"passed": passed, "checks": checks, "metrics": m, "thresholds": g})
print(f"[07] quality gate {'PASSED' if passed else 'FAILED'} {checks}")
sys.exit(0 if passed else 1)
