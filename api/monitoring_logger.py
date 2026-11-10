import json
import time
from datetime import datetime, timezone
from pathlib import Path

LOG = Path(__file__).resolve().parents[1] / "logs" / "inference" / "inference_log.jsonl"


def log_request(endpoint: str, n_rows: int, mean_prediction: float, started: float, status: str = "ok"):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "endpoint": endpoint, "rows": n_rows,
                            "mean_prediction": round(mean_prediction, 2),
                            "latency_ms": round((time.time() - started) * 1000, 2), "status": status}) + "\n")
