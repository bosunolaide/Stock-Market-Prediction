from __future__ import annotations

from sklearn.utils.validation import check_is_fitted

from dataclasses import dataclass
from typing import Optional, Tuple

import pickle
import tensorflow as tf

from .logging_config import setup_logging
from .mlflow_utils import download_registered_artifacts
from .settings import settings

logger = setup_logging(__name__)


class _CompatUnpickler(pickle.Unpickler):
    """Compatibility unpickler for legacy pickles.

    Some legacy artifacts were pickled with MultiDimensionScaler defined in __main__.
    We remap that symbol to stock_mlops.scalers.MultiDimensionScaler so older files load.
    """

    def find_class(self, module, name):
        if module == "__main__" and name == "MultiDimensionScaler":
            from .scalers import MultiDimensionScaler
            return MultiDimensionScaler
        return super().find_class(module, name)

def _pickle_load_compat(file_obj):
    return _CompatUnpickler(file_obj).load()

def _ensure_feature_scaler_fitted(scaler_obj: object, name: str) -> None:
    """
    Notebook-style MultiDimensionScaler stores a list of sklearn scalers in `.scalers`.
    """
    scalers = getattr(scaler_obj, "scalers", None)
    if not scalers or not isinstance(scalers, (list, tuple)):
        raise RuntimeError(f"{name} has no `.scalers` list (wrong artifact type?)")
    for i, s in enumerate(scalers):
        try:
            check_is_fitted(s)
        except Exception as e:
            raise RuntimeError(f"{name}.scalers[{i}] is not fitted") from e

def _ensure_target_scaler_fitted(scaler_obj: object, name: str) -> None:
    """
    Target scaler in your notebook is a plain MinMaxScaler.
    """
    try:
        check_is_fitted(scaler_obj)
    except Exception as e:
        raise RuntimeError(f"{name} is not fitted") from e

@dataclass
class ModelBundle:
    model: tf.keras.Model
    feature_scaler: object
    target_scaler: object
    source: str  # "mlflow" or "fallback"

def _load_fallback() -> ModelBundle:
    model = tf.keras.models.load_model(settings.fallback_model_path)
    with open(settings.fallback_feature_scaler_path, "rb") as f:
        feature_scaler = _pickle_load_compat(f)
    with open(settings.fallback_target_scaler_path, "rb") as f:
        target_scaler = _pickle_load_compat(f)

    _ensure_feature_scaler_fitted(feature_scaler, "feature_scaler (fallback)")
    _ensure_target_scaler_fitted(target_scaler, "target_scaler (fallback)")

    return ModelBundle(model=model, feature_scaler=feature_scaler, target_scaler=target_scaler, source="fallback")



def load_model_bundle() -> ModelBundle:
    """Load model + scalers from MLflow Model Registry if available; otherwise fall back to repo artifacts."""
    try:
        artifacts = download_registered_artifacts()
        if artifacts:
            logger.info("Loading model bundle from MLflow artifacts (%s).", artifacts.model_uri)
            model = tf.keras.models.load_model(artifacts.model_path)

            with open(artifacts.feature_scaler_path, "rb") as f:
                feature_scaler = _pickle_load_compat(f)

            with open(artifacts.target_scaler_path, "rb") as f:
                target_scaler = _pickle_load_compat(f)

            _ensure_feature_scaler_fitted(feature_scaler, "feature_scaler (mlflow)")
            _ensure_target_scaler_fitted(target_scaler, "target_scaler (mlflow)")

            return ModelBundle(
                model=model,
                feature_scaler=feature_scaler,
                target_scaler=target_scaler,
                source="mlflow",
            )


    except Exception as e:
        logger.warning("MLflow load failed, using fallback. Error: %s", e)

    logger.info("Loading model bundle from fallback local artifacts.")
    return _load_fallback()
