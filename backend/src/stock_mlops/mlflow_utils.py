from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import mlflow
from mlflow.tracking import MlflowClient

from .logging_config import setup_logging
from .settings import settings

logger = setup_logging(__name__)

@dataclass
class LoadedArtifacts:
    model_path: str
    feature_scaler_path: str
    target_scaler_path: str
    run_id: Optional[str] = None
    model_uri: Optional[str] = None

def configure_mlflow() -> None:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", settings.mlflow_tracking_uri))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", settings.mlflow_experiment))

def get_client() -> MlflowClient:
    configure_mlflow()
    return MlflowClient()

def get_model_version_uri(model_name: str, stage: str | None = None) -> Optional[str]:
    client = get_client()
    if stage:
        versions = client.get_latest_versions(model_name, stages=[stage])
        if versions:
            v = versions[0]
            return f"models:/{model_name}/{stage}"
    # fallback to latest version in registry
    try:
        latest = client.search_model_versions(f"name='{model_name}'", max_results=1, order_by=["version_number DESC"])
        if latest:
            v = latest[0]
            return f"models:/{model_name}/{v.version}"
    except Exception:
        return None
    return None

def download_registered_artifacts(
    model_name: str | None = None,
    stage: str | None = None,
    dst_dir: str = "artifacts/mlflow",
) -> Optional[LoadedArtifacts]:
    model_name = model_name or settings.mlflow_registered_model_name
    stage = stage if stage is not None else settings.mlflow_stage

    model_uri = get_model_version_uri(model_name, stage=stage)
    if not model_uri:
        logger.warning("No MLflow registered model found (name=%s, stage=%s).", model_name, stage)
        return None

    configure_mlflow()
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    # We expect artifacts logged under a folder "model" plus scaler files.
    # This function downloads the full model directory plus scalers if present.
    local_model_dir = mlflow.artifacts.download_artifacts(artifact_uri=f"{model_uri}/model", dst_path=str(dst))
    feature_scaler = mlflow.artifacts.download_artifacts(artifact_uri=f"{model_uri}/Feature_Scaler.pck", dst_path=str(dst))
    target_scaler = mlflow.artifacts.download_artifacts(artifact_uri=f"{model_uri}/Target_Scaler.pck", dst_path=str(dst))

    return LoadedArtifacts(
        model_path=str(Path(local_model_dir)),
        feature_scaler_path=str(Path(feature_scaler)),
        target_scaler_path=str(Path(target_scaler)),
        model_uri=model_uri,
    )
