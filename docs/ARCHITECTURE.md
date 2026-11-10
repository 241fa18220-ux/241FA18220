# Architecture
country_wise_latest.csv → validate → preprocess (impute + scale + one-hot) → 4 regressors compared by 5-fold CV RMSE
→ best model → evaluate on held-out 20% → quality gate (R², RMSE vs mean baseline) → deployment/ → FastAPI → Docker → K8s.
Monitoring compares new data to the training baseline (PSI) and re-scores labeled new data.
