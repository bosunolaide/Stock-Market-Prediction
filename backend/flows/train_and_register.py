from __future__ import annotations

from pathlib import Path

import mlflow
from prefect import flow, task

from stock_mlops.logging_config import setup_logging
from stock_mlops.mlflow_utils import configure_mlflow
from stock_mlops.settings import settings

logger = setup_logging(__name__)

@task(retries=2, retry_delay_seconds=10)
def log_and_register_existing_artifacts() -> str:
    """
    This repo currently ships with trained artifacts:
      - Model.h5
      - Feature_Scaler.pck
      - Target_Scaler.pck

    To make the project production-grade immediately, this task logs them to MLflow
    and registers the run as a versioned model in the Model Registry.

    Later, you can replace this with a real re-training step.
    """
    configure_mlflow()

    model_path = Path(settings.fallback_model_path)
    feature_scaler = Path(settings.fallback_feature_scaler_path)
    target_scaler = Path(settings.fallback_target_scaler_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model artifact: {model_path}")
    if not feature_scaler.exists():
        raise FileNotFoundError(f"Missing scaler artifact: {feature_scaler}")
    if not target_scaler.exists():
        raise FileNotFoundError(f"Missing scaler artifact: {target_scaler}")

    with mlflow.start_run(run_name="register-existing-artifacts") as run:
        mlflow.log_param("source", "existing_repo_artifacts")

        # Log Keras model directory under 'model'
        mlflow.keras.log_model(
            keras_model=str(model_path),
            artifact_path="model",
            registered_model_name=settings.mlflow_registered_model_name,
        )

        mlflow.log_artifact(str(feature_scaler), artifact_path="")
        mlflow.log_artifact(str(target_scaler), artifact_path="")

        run_id = run.info.run_id
        logger.info("Logged and registered model artifacts. run_id=%s", run_id)
        return run_id

@flow(name="train-and-register")
def train_and_register_flow():
    run_id = log_and_register_existing_artifacts()
    logger.info("Done. MLflow run: %s", run_id)

if __name__ == "__main__":
    train_and_register_flow()
