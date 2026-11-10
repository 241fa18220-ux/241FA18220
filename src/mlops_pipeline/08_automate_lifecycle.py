"""Step 8 - run the whole lifecycle (01-07) and record the result."""
import subprocess
import sys
import time
from pathlib import Path

from validation import write_json

here = Path(__file__).parent
steps = ["01_load_data.py", "02_validate_data.py", "03_preprocessing.py",
         "04_validate_outputs.py", "05_train_registry.py",
         "06_model_evaluation.py", "07_quality_gate.py"]
log, ok = [], True
for s in steps:
    t = time.time()
    rc = subprocess.call([sys.executable, str(here / s)])
    log.append({"step": s, "returncode": rc, "seconds": round(time.time() - t, 2)})
    if rc != 0:
        ok = False
        break
write_json("logs/pipeline_run_log.json", {"success": ok, "steps": log})
write_json("reports/lifecycle_report.json", {"success": ok, "steps": log})
print("[08] lifecycle", "SUCCESS" if ok else "FAILED")
sys.exit(0 if ok else 1)
