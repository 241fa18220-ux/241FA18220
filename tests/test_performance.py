import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_quality_gate_passed():
    assert json.loads((ROOT / "reports/quality_gate_report.json").read_text())["passed"]


def test_model_beats_mean_baseline():
    m = json.loads((ROOT / "reports/evaluation_report.json").read_text())
    assert m["rmse"] < m["baseline_rmse"]
