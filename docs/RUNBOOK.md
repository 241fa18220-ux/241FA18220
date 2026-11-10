# Runbook
- API returns 503 → `deployment/model.pkl` missing: run `11_prepare_deployment.py`.
- Quality gate failed → see `reports/quality_gate_report.json`; improve the model or revisit thresholds in `config.yaml`.
- Retrain triggered → run `12_retrain_model.py`, check `reports/model_comparison.json`, redeploy.
