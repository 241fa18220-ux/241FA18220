"""Step 11 - copy approved model artifacts into deployment/."""
import shutil
from datetime import datetime, timezone

from validation import ROOT, load_config, path, read_json, write_json

cfg = load_config()
gate = read_json("reports/quality_gate_report.json")
if not gate["passed"]:
    raise SystemExit("[11] quality gate not passed - refusing to deploy")
m, d = ROOT / cfg["paths"]["models_dir"], cfg["paths"]["deployment_dir"]
shutil.copy(m / "covid_model.pkl", path(f"{d}/model.pkl"))
shutil.copy(m / "preprocessor.pkl", path(f"{d}/preprocessor.pkl"))
write_json(f"{d}/model_metadata.json", {
    "deployed_at": datetime.now(timezone.utc).isoformat(), "metrics": gate["metrics"],
    "model_report": read_json("artifacts/production_model_report.json")})
print("[11] deployment/ updated")
