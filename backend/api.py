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
    Feature_Scaler = bundle.feature_scaler
    Target_Scaler = bundle.target_scaler
    if model is None or Feature_Scaler is None or Target_Scaler is None:
        raise HTTPException(status_code=500, detail="Model or scalers not loaded")
    df.index = pd.to_datetime(df.index)
    idx_location = df.index.get_loc(previous_date.strftime("%Y-%m-%d"), method="pad")
    Features = df.iloc[idx_location - feature_length: idx_location, :].values
    Features = np.expand_dims(Features, axis=0)
    Features = Feature_Scaler.transform(Features)
    Prediction = model.predict(Features)
    Prediction = Target_Scaler.inverse_transform(Prediction)
    return float(Prediction[0][0])

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
    return {"status": "ok", "model_loaded": Model is not None}
