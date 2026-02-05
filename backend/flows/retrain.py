from __future__ import annotations

import json
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
from prefect import flow, task

from stock_mlops.logging_config import setup_logging
from stock_mlops.settings import settings
from stock_mlops.training import TrainConfig, download_history, train_end_to_end
from stock_mlops.validation import validate_yfinance_frame
from stock_mlops.drift import run_evidently_drift

logger = setup_logging(__name__)


@task
def fetch_data(cfg: TrainConfig):
    df = download_history(cfg.ticker, cfg.start, cfg.end)
    return df


@task
def validate_data(df):
    return validate_yfinance_frame(df)


@task
def drift_check(df, drift_days: int = 60):
    # simple reference/current split by time: last N rows as current
    if len(df) <= drift_days + 10:
        return {"success": True, "summary": {"skipped": "not enough rows"}, "report_path": None}

    reference = df.iloc[:-drift_days].reset_index()
    current = df.iloc[-drift_days:].reset_index()

    res = run_evidently_drift(reference=reference, current=current, report_dir="artifacts/drift")
    return {"success": res.success, "summary": res.summary, "report_path": res.report_path}


@task
def train(cfg: TrainConfig):
    return train_end_to_end(cfg)


@task
def log_and_register(model, feature_scaler, target_scaler, metrics: dict, artifacts: dict, drift: dict, validation: dict):
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)

    params = {
        "ticker": metrics.get("ticker"),
        "start": metrics.get("start"),
        "end": metrics.get("end"),
        "feature_length": metrics.get("feature_length"),
        "epochs": metrics.get("epochs"),
        "batch_size": artifacts.get("batch_size", None),
    }

    with mlflow.start_run(run_name=f"retrain-{metrics.get('ticker')}") as run:
        run_id = run.info.run_id

        # params + metrics
        for k, v in params.items():
            if v is not None:
                mlflow.log_param(k, v)
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, float(v))

        # log json artifacts
        Path("artifacts").mkdir(exist_ok=True)
        meta = {
            "validation": validation,
            "drift": drift,
            "metrics": metrics,
        }
        meta_path = Path("artifacts") / "run_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        mlflow.log_artifact(str(meta_path), artifact_path="reports")

        if drift.get("report_path"):
            mlflow.log_artifact(drift["report_path"], artifact_path="reports")

        # log scalers
        scaler_dir = Path("artifacts/scalers")
        scaler_dir.mkdir(parents=True, exist_ok=True)

        import pickle
        fs_path = scaler_dir / "Feature_Scaler.pck"
        ts_path = scaler_dir / "Target_Scaler.pck"
        with open(fs_path, "wb") as f:
            pickle.dump(feature_scaler, f)
        with open(ts_path, "wb") as f:
            pickle.dump(target_scaler, f)

        mlflow.log_artifact(str(fs_path), artifact_path="")
        mlflow.log_artifact(str(ts_path), artifact_path="")

        # log keras model
        
# log keras model (as MLflow model + register)
mlflow.keras.log_model(
    model,
    artifact_path="model",
    registered_model_name=settings.mlflow_registered_model_name,
)

        logger.info("Logged + registered model to MLflow (run_id=%s).", run_id)

    # Promote latest version to Production (optional, best-effort)
    try:
        client = MlflowClient()
        versions = client.get_latest_versions(settings.mlflow_registered_model_name)
        if versions:
            latest = sorted(versions, key=lambda v: int(v.version))[-1]
            client.transition_model_version_stage(
                name=settings.mlflow_registered_model_name,
                version=latest.version,
                stage="Production",
                archive_existing_versions=True,
            )
            logger.info("Promoted model %s v%s to Production.", settings.mlflow_registered_model_name, latest.version)
    except Exception as e:
        logger.warning("Could not promote model to Production: %s", e)

    return {"status": "ok", "model": settings.mlflow_registered_model_name}


@flow(name="true_retrain_pipeline")
def true_retrain_pipeline(
    ticker: str = "AAPL",
    start: str = "2015-01-01",
    end: str = "2025-01-01",
    feature_length: int = 32,
    epochs: int = 10,
    batch_size: int = 64,
):
    cfg = TrainConfig(ticker=ticker, start=start, end=end, feature_length=feature_length, epochs=epochs, batch_size=batch_size)

    df = fetch_data(cfg)
    val = validate_data(df)
    if not val.success:
        raise ValueError(f"Data validation failed: {val.summary}")

    drift = drift_check(df)

    model, feature_scaler, target_scaler, metrics, artifacts = train(cfg)
    result = log_and_register(model, feature_scaler, target_scaler, metrics, artifacts, drift, val.summary)
    return result


if __name__ == "__main__":
    true_retrain_pipeline()
