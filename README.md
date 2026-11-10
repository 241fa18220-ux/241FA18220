# COVID-19 Recovery Prediction – End-to-End MLOps Project

Predicts the number of **Recovered** COVID-19 cases per country from `country_wise_latest.csv`
(regression). Covers the full lifecycle: data validation → preprocessing → training/tracking → evaluation →
quality gate → deployment (FastAPI + Docker + Kubernetes) → monitoring, drift detection and retraining.

The exploration / modelling notebook is in `notebooks/covid_recovery.ipynb`; the pipeline scripts reproduce its logic
(same dropped columns, same leakage removal, 80/20 split with seed 42, same 4 models).

## Quick start
```bash
python -m venv .venv && .venv\Scripts\activate        # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt -r requirements-api.txt

python src/mlops_pipeline/08_automate_lifecycle.py    # runs steps 01-07
python src/make_demo_new_data.py                      # demo file for batch inference (optional)
python src/mlops_pipeline/11_prepare_deployment.py    # copies model into deployment/
uvicorn api.main:app --reload                         # http://localhost:8000/docs
pytest -q
```

## Data & features
| Role | Columns |
|---|---|
| Target | `Recovered` |
| Numeric features | Confirmed, Deaths, New cases, New deaths, New recovered, Deaths / 100 Cases, Confirmed last week |
| Categorical | WHO Region |
| Identifier (not a feature) | Country/Region |
| Dropped (unused) | 1 week change, 1 week % increase |
| Dropped (target leakage) | Recovered / 100 Cases, Deaths / 100 Recovered, Active |

## Pipeline steps (`src/mlops_pipeline/`)
| Step | Script | Output |
|---|---|---|
| 01 | `01_load_data.py` | `artifacts/data_load_report.json` |
| 02 | `02_validate_data.py` | `artifacts/baseline_validation.csv` |
| 03 | `03_preprocessing.py` | `data/processed/*`, `models/preprocessor.pkl` |
| 04 | `04_validate_outputs.py` | shape / NaN checks |
| 05 | `05_train_registry.py` | CV-compares Linear / Decision Tree / Random Forest / Gradient Boosting; saves best |
| 06 | `06_model_evaluation.py` | `reports/evaluation_report.json`, actual-vs-predicted + residual plots |
| 07 | `07_quality_gate.py` | `reports/quality_gate_report.json` (exit 1 on fail) |
| 08 | `08_automate_lifecycle.py` | runs 01–07 |
| 09 | `09_inference.py` | batch predictions for `data/raw/new_countries.csv` |
| 10 | `10_failure_injection.py` | proves validation catches corrupted data |
| 11 | `11_prepare_deployment.py` | `deployment/` (only if the gate passed) |
| 12 | `12_*.py` | evolved data, data/prediction drift, performance monitoring, retrain trigger, retraining |

## API
`GET /health` · `POST /predict` · `POST /predict_batch` (see `/docs`). Example body:
```json
{"country":"Testland","confirmed":50000,"deaths":1200,"new_cases":400,"new_deaths":10,
 "new_recovered":300,"deaths_per_100_cases":2.4,"confirmed_last_week":47000,"who_region":"Europe"}
```

## Docker & Kubernetes
```bash
docker build -t covid-api .
docker run -p 8000:8000 covid-api
kubectl apply -f k8s/            # NodePort 30080
```

## Notes
- Only 187 countries → the test set is 38 rows, so metrics are noisy and sensitive to outliers (USA, Brazil, India).
- Quality-gate thresholds live in `config.yaml`.
