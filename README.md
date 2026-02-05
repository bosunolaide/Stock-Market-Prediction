# 📈 Stock Market Prediction — Production‑Grade MLOps Upgrade

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?logo=mlflow)
![Prefect](https://img.shields.io/badge/Prefect-0B0F19?logo=prefect)
![Prometheus](https://img.shields.io/badge/Prometheus-E6522C?logo=prometheus)
![Grafana](https://img.shields.io/badge/Grafana-F46800?logo=grafana)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

This repo started as a **FastAPI + Streamlit** stock prediction app.  
It has now been upgraded into an **end‑to‑end production‑style MLOps system**:

- ✅ **MLflow Tracking + Model Registry** (versioned models, promotion to Production)
- ✅ **Object storage** (MinIO S3 bucket for artifacts; compatible with AWS S3)
- ✅ **Pipeline orchestration** with **Prefect** (register model, daily inference, evaluation)
- ✅ **Monitoring** with **Prometheus + Pushgateway + Grafana** (model RMSE tracked over time)
- ✅ **Makefile automation**, structured `src/` package layout, logging, and tests
- ✅ Backend API now loads the **Production model from MLflow** (fallback to local artifacts if registry is empty)

> Note: the “training” flow currently **registers the existing `Model.h5` + scalers** shipped with this repo.
> You can swap it for a real re-training step later without changing the rest of the production plumbing.

---

## 🧱 Architecture

**Services (Docker Compose):**
- `backend` — FastAPI inference API
- `frontend` — Streamlit UI
- `mlflow` — MLflow Tracking Server + Model Registry
- `postgres` — MLflow backend store
- `minio` — S3-compatible artifact store
- `prometheus` + `pushgateway` — metrics collection
- `grafana` — monitoring dashboards

---

## ⚡ Quickstart (Local)

### 1) Start the full stack

```bash
make up
```

Key URLs:
- FastAPI: http://localhost:8000/docs
- Streamlit: http://localhost:8501
- MLflow: http://localhost:5000
- Grafana: http://localhost:3000  (default login: admin / admin)
- MinIO Console: http://localhost:9001  (minioadmin / minioadmin)

### 2) Register the model in MLflow (first time)

This logs the existing artifacts to MLflow and creates a registered model version.

```bash
make register-model
```

### 3) Run daily batch inference

```bash
make daily-inference
```

This writes a JSON forecast to:

- `backend/artifacts/forecasts/latest_forecast.json`

### 4) Evaluate + publish monitoring metrics

```bash
make evaluate
```

This computes a simple RMSE on recent data and pushes it to Pushgateway.
Grafana dashboard: **Stock Forecast - Model Monitoring**.

---

## 🔁 How model loading works (API)

The backend loads in this order:

1. Try MLflow Model Registry: `models:/stock_forecast_model/Production`
2. If registry is empty/unavailable → fall back to local repo artifacts:
   - `Model.h5`
   - `Feature_Scaler.pck`
   - `Target_Scaler.pck`

You can reload the model without redeploy:

- `POST /reload-model`

---

## 🧪 Tests

```bash
make test
```

---

## 📦 Repo layout (new pieces)

```
backend/
  src/stock_mlops/          # reusable MLOps utilities
  flows/                    # Prefect flows (register, inference, evaluate)
  artifacts/                # outputs (forecasts, downloaded mlflow artifacts, etc.)
monitoring/
  prometheus/
  grafana/
```

---

## 🔒 Production notes

When you move from MinIO to real AWS S3:

- Set:
  - `S3_ENDPOINT_URL` (leave empty for AWS)
  - `AWS_ACCESS_KEY_ID`
  - `AWS_SECRET_ACCESS_KEY`
  - `S3_BUCKET`
- Update MLflow artifact root accordingly.

---

## License

MIT


## True re-training pipeline (yfinance → validate → drift → train → register)

This repo includes a production-style re-training flow that:

1) Downloads historical OHLCV data via `yfinance`  
2) Validates the dataset with **Great Expectations** (schema + basic sanity checks)  
3) Runs an **Evidently** data drift report (reference vs recent window)  
4) Trains a sequence model end-to-end  
5) Logs metrics + artifacts to **MLflow** and registers the model  
6) Attempts to promote the latest model version to **Production**

### Run it locally

Start the stack:

```bash
make up
```

Then trigger retraining:

```bash
make retrain
```

You can also run with custom parameters inside the backend container:

```bash
docker compose exec backend python flows/retrain.py
```

Drift report (HTML) is saved under `backend/artifacts/drift/` and also logged to MLflow.

## CI/CD

GitHub Actions workflows are included:

- `CI` runs ruff + pytest on PRs and pushes
- `Build & Push Docker Images` builds and pushes backend/frontend images to GHCR

To publish images, ensure GitHub Packages permissions are enabled for your repo/organization.
