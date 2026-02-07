from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import tensorflow as tf
import pickle
import pandas as pd
import numpy as np
import yfinance as yf

from stock_mlops.model_loader import load_model_bundle
from stock_mlops.logging_config import setup_logging

from datetime import datetime, timedelta

app = FastAPI(title="Stock Market Prediction API")

logger = setup_logging(__name__)
bundle = None

@app.on_event("startup")
def _startup_load_model():
    global bundle
    bundle = load_model_bundle()
    logger.info("Loaded model bundle (source=%s).", getattr(bundle, "source", "unknown"))

@app.post("/reload-model")
def reload_model():
    """Reload model bundle from MLflow registry (or fallback) without redeploying."""
    global bundle
    bundle = load_model_bundle()
    return {"status": "ok", "source": getattr(bundle, "source", "unknown")}


class MultiDayPredictionRequest(BaseModel):
    ticker: str
    start_date: datetime
    end_date: datetime
    previous_date: datetime
    days: int = 7
    feature_length: int = 32

class ForecastPoint(BaseModel):
    date: datetime
    predicted_price: float

class MultiDayPredictionResponse(BaseModel):
    ticker: str
    start_date: datetime
    previous_date: datetime
    days: int
    forecast: List[ForecastPoint]

def predict_one_day(model, df: pd.DataFrame, previous_date: datetime, feature_length: int = 32) -> float:
    if bundle is None:
        raise HTTPException(status_code=500, detail="Model bundle not loaded")

    feature_scaler = bundle.feature_scaler
    target_scaler = bundle.target_scaler

    if model is None or feature_scaler is None or target_scaler is None:
        raise HTTPException(status_code=500, detail="Model or scalers not loaded")

    # Ensure DatetimeIndex and sorted ascending (required for pad/ffill indexing)
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Convert previous_date to a pandas Timestamp (naive)
    ts = pd.Timestamp(previous_date)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)

    # Find the nearest index at-or-before previous_date (pad / forward-fill)
    idx = df.index.get_indexer([ts], method="pad")[0]
    if idx == -1:
        raise HTTPException(
            status_code=400,
            detail=f"previous_date={ts.date()} is earlier than the first available date {df.index.min().date()}",
        )

    if idx < feature_length:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough historical rows before {ts.date()} to build a window of length {feature_length}. "
                   f"Need at least {feature_length} rows, but have {idx}.",
        )

    # Build feature window
    features = df.iloc[idx - feature_length: idx, :].values  # shape: (feature_length, n_features)
    features = np.expand_dims(features, axis=0)              # shape: (1, feature_length, n_features)

    # Scale + predict + inverse scale
    features = feature_scaler.transform(features)
    prediction = model.predict(features)
    prediction = target_scaler.inverse_transform(prediction)

    return float(prediction[0][0])


def predict_multiple_days(model, df, previous_date, days=7, feature_length=32):
    df_copy = df.copy()
    df_copy.index = pd.to_datetime(df_copy.index)
    preds, dates = [], []
    current_date = previous_date
    for _ in range(days):
        pred = predict_one_day(model, df_copy, current_date, feature_length)
        next_date = current_date + timedelta(days=1)
        new_row = df_copy.iloc[-1:].copy()
        new_row.index = [next_date]
        new_row["Close"] = pred
        df_copy = pd.concat([df_copy, new_row])
        preds.append(pred)
        dates.append(next_date)
        current_date = next_date
    return dates, preds

@app.post("/predict/multi-day", response_model=MultiDayPredictionResponse)
def predict_multi_day(req: MultiDayPredictionRequest):
    df = yf.download(req.ticker, start=req.start_date, end=req.end_date)
    if df.empty:
        raise HTTPException(status_code=400, detail="No data returned for ticker/date range")
    dates, preds = predict_multiple_days(bundle.model, df, req.previous_date, req.days, req.feature_length)
    forecast = [ForecastPoint(date=d, predicted_price=p) for d, p in zip(dates, preds)]
    return MultiDayPredictionResponse(
        ticker=req.ticker, start_date=req.start_date, previous_date=req.previous_date, days=req.days, forecast=forecast
    )

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bundle is not None}
