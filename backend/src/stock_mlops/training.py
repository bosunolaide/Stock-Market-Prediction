from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from .logging_config import setup_logging
from .scalers import MultiDimensionScaler

logger = setup_logging(__name__)

DEFAULT_FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
DEFAULT_TARGET_COLUMN = "Close"


@dataclass
class TrainConfig:
    ticker: str = "AAPL"
    start: str = "2015-01-01"
    end: str = "2025-01-01"
    feature_length: int = 32
    test_size: float = 0.2
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    seed: int = 42


def download_history(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No data returned for ticker={ticker} start={start} end={end}")
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    return df


def make_sequences(
    df: pd.DataFrame,
    feature_length: int,
    feature_columns: Optional[List[str]] = None,
    target_column: str = DEFAULT_TARGET_COLUMN,
) -> Tuple[np.ndarray, np.ndarray]:
    feature_columns = feature_columns or DEFAULT_FEATURE_COLUMNS
    if any(c not in df.columns for c in feature_columns):
        missing = [c for c in feature_columns if c not in df.columns]
        raise ValueError(f"Missing feature columns: {missing}. df.columns={list(df.columns)}")
    if target_column not in df.columns:
        raise ValueError(f"Missing target column: {target_column}")

    X_list, y_list = [], []
    values = df[feature_columns].values.astype(np.float32)
    target = df[[target_column]].values.astype(np.float32)

    for i in range(feature_length, len(df)):
        X_list.append(values[i - feature_length : i, :])
        y_list.append(target[i, :])

    X = np.stack(X_list, axis=0)  # (samples, timesteps, features)
    y = np.stack(y_list, axis=0)  # (samples, 1)
    return X, y


def train_test_split_time(X: np.ndarray, y: np.ndarray, test_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(X)
    n_test = int(np.ceil(n * test_size))
    n_train = n - n_test
    return X[:n_train], X[n_train:], y[:n_train], y[n_train:]


def build_model(input_shape: Tuple[int, int], learning_rate: float) -> tf.keras.Model:
    # input_shape: (timesteps, features)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    return model


def fit_scalers(X_train: np.ndarray, y_train: np.ndarray):
    feature_scaler = MultiDimensionScaler().fit(X_train)
    target_scaler = MinMaxScaler().fit(y_train)
    return feature_scaler, target_scaler


def train_end_to_end(cfg: TrainConfig):
    """True re-training pipeline: download -> sequence -> scale -> train -> evaluate.

    Returns:
        model, feature_scaler, target_scaler, metrics dict, and a dict of sample artifacts.
    """
    tf.random.set_seed(cfg.seed)
    np.random.seed(cfg.seed)

    df = download_history(cfg.ticker, cfg.start, cfg.end)
    X, y = make_sequences(df, cfg.feature_length)

    X_train, X_test, y_train, y_test = train_test_split_time(X, y, cfg.test_size)

    feature_scaler, target_scaler = fit_scalers(X_train, y_train)

    X_train_s = feature_scaler.transform(X_train)
    X_test_s = feature_scaler.transform(X_test)

    y_train_s = target_scaler.transform(y_train)
    y_test_s = target_scaler.transform(y_test)

    model = build_model(input_shape=(X_train_s.shape[1], X_train_s.shape[2]), learning_rate=cfg.learning_rate)

    history = model.fit(
        X_train_s,
        y_train_s,
        validation_data=(X_test_s, y_test_s),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        verbose=1,
    )

    # Predict and compute RMSE in original scale
    y_pred_s = model.predict(X_test_s, verbose=0)
    y_pred = target_scaler.inverse_transform(y_pred_s)
    y_true = y_test

    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    metrics = {
        "rmse": rmse,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "feature_length": int(cfg.feature_length),
        "ticker": cfg.ticker,
        "start": cfg.start,
        "end": cfg.end,
        "epochs": int(cfg.epochs),
    }

    artifacts = {
        "history": history.history,
        "y_pred_head": y_pred[:10].reshape(-1).tolist(),
        "y_true_head": y_true[:10].reshape(-1).tolist(),
    }

    logger.info("Training done: rmse=%.6f (ticker=%s)", rmse, cfg.ticker)
    return model, feature_scaler, target_scaler, metrics, artifacts
