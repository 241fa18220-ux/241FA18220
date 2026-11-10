"""Deploy only if the quality gate passed."""
import subprocess
import sys
from pathlib import Path

from validation import read_json

if read_json("reports/quality_gate_report.json")["passed"]:
    sys.exit(subprocess.call([sys.executable, str(Path(__file__).parent / "11_prepare_deployment.py")]))
print("Not promoted - skipping deployment")
