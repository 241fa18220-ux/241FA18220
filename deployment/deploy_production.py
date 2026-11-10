"""Production deploy entry point: run the lifecycle, then deploy if promoted."""
import subprocess
import sys
from pathlib import Path

steps = Path(__file__).resolve().parents[1] / "src" / "mlops_pipeline"
for s in ("08_automate_lifecycle.py", "deploy_if_promoted.py"):
    rc = subprocess.call([sys.executable, str(steps / s)])
    if rc:
        sys.exit(rc)
print("Deployment artifacts ready in deployment/")
