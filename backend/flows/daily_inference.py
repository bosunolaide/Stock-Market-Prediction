from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
from prefect import flow, task

from stock_mlops.logging_config import setup_logging
from stock_mlops.model_loader import load_model_bundle

logger = setup_logging(__name__)

@task(retries=2, retry_delay_seconds=10)
def fetch_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end)
    if df is None or df.empty:
        raise ValueError("No data returned from yfinance.")
    # Keep same columns used by your existing model (assumes OHLCV)
    return df

@task
def predict_next_days(ticker: str, df: pd.DataFrame, days: int = 7, feature_length: int = 32) -> dict:
    bundle = load_model_bundle()
    model, feature_scaler, target_scaler = bundle.model, bundle.feature_scaler, bundle.target_scaler
    logger.info("Using model source: %s", bundle.source)

    df = df.copy()
    df.index = pd.to_datetime(df.index)

    # Use last available day as "previous_date"
    previous_date = df.index.max()

    preds = []
    current_date = previous_date

    df_copy = df.copy()
    import numpy as np

    for i in range(days):
        # feature window
        window = df_copy.iloc[-feature_length:, :].values
        x = np.expand_dims(window, axis=0)
        x = feature_scaler.transform(x)
        y = model.predict(x, verbose=0)
        y_inv = target_scaler.inverse_transform(y)[0][0]
        current_date = current_date + timedelta(days=1)
        preds.append({"date": current_date.isoformat(), "predicted_price": float(y_inv)})

        # naive append row to keep rolling window (uses predicted close only; you can improve later)
        new_row = df_copy.iloc[-1].copy()
        if "Close" in df_copy.columns:
            new_row["Close"] = y_inv
        df_copy.loc[current_date] = new_row

    return {"ticker": ticker, "generated_at": datetime.utcnow().isoformat(), "forecast": preds}

@task
def save_forecast(result: dict, out_path: str = "artifacts/forecasts/latest_forecast.json") -> str:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return str(p)

@flow(name="daily-inference")
def daily_inference_flow(ticker: str = "AAPL", days: int = 7, lookback_days: int = 365, feature_length: int = 32):
    end = datetime.utcnow().date().isoformat()
    start = (datetime.utcnow().date() - timedelta(days=lookback_days)).isoformat()
    df = fetch_data(ticker, start, end)
    result = predict_next_days(ticker, df, days=days, feature_length=feature_length)
    path = save_forecast(result)
    logger.info("Saved forecast to %s", path)

if __name__ == "__main__":
    daily_inference_flow()
