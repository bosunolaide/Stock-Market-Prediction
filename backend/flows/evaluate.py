from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from prefect import flow, task

from stock_mlops.logging_config import setup_logging
from stock_mlops.model_loader import load_model_bundle
from stock_mlops.monitoring import push_rmse

logger = setup_logging(__name__)

@task
def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end)
    if df is None or df.empty:
        raise ValueError("No data returned from yfinance.")
    return df

@task
def compute_rmse(ticker: str, df: pd.DataFrame, horizon_days: int = 7, feature_length: int = 32) -> float:
    # Simple backtest: predict next day for last horizon_days and compare to actual closes
    bundle = load_model_bundle()
    model, feature_scaler, target_scaler = bundle.model, bundle.feature_scaler, bundle.target_scaler
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    if "Close" not in df.columns:
        raise ValueError("Data must contain Close column.")

    y_true = []
    y_pred = []

    for i in range(horizon_days):
        idx = len(df) - horizon_days + i
        window = df.iloc[idx-feature_length:idx, :].values
        x = np.expand_dims(window, axis=0)
        x = feature_scaler.transform(x)
        pred = model.predict(x, verbose=0)
        pred_close = float(target_scaler.inverse_transform(pred)[0][0])
        y_pred.append(pred_close)
        y_true.append(float(df.iloc[idx]["Close"]))

    rmse = float(np.sqrt(np.mean((np.array(y_pred) - np.array(y_true)) ** 2)))
    logger.info("RMSE over last %d days = %.6f", horizon_days, rmse)
    return rmse

@task
def push_metrics(rmse: float) -> None:
    push_rmse(rmse)

@flow(name="evaluate-and-monitor")
def evaluate_flow(ticker: str = "AAPL", horizon_days: int = 7, lookback_days: int = 365, feature_length: int = 32):
    end = datetime.utcnow().date().isoformat()
    start = (datetime.utcnow().date() - timedelta(days=lookback_days)).isoformat()
    df = fetch_data(ticker, start, end)
    rmse = compute_rmse(ticker, df, horizon_days=horizon_days, feature_length=feature_length)
    push_metrics(rmse)

if __name__ == "__main__":
    evaluate_flow()
