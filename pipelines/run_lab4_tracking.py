import subprocess, sys
from pathlib import Path
sys.exit(subprocess.call([sys.executable, str(Path(__file__).resolve().parents[1] / "src" / "mlops_pipeline" / "05_train_registry.py")]))
