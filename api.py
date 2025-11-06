from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import tensorflow as tf
import pickle
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

app = FastAPI(title="Stock Market Prediction API")

# ============================
# Load model & scalers at startup
# ============================
try:
    Model = tf.keras.models.load_model("Model.h5")
    with open("Feature_Scaler.pck", "rb") as f:
        Feature_Scaler = pickle.load(f)
    with open("Target_Scaler.pck", "rb") as f:
        Target_Scaler = pickle.load(f)
except Exception as e:
    # If this fails, the API will still start but raise on first request
    Model = None
    Feature_Scaler = None
    Target_Scaler = None
    print("❌ Error loading model/scalers:", e)


# ============================
# Pydantic request/response models
# ============================
class SinglePredictionRequest(BaseModel):
    ticker: str
    start_date: datetime
    end_date: datetime
    previous_date: datetime
    feature_length: int = 32


class SinglePredictionResponse(BaseModel):
    ticker: str
    previous_date: datetime
    next_date: datetime
    predicted_price: float


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


# ============================
# Core prediction functions (same logic as Streamlit)
# ============================
def predict_one_day(model, df: pd.DataFrame, previous_date: datetime, feature_length: int = 32) -> float:
    if model is None or Feature_Scaler is None or Target_Scaler is None:
        raise HTTPException(status_code=500, detail="Model or scalers not loaded")

    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    prev_str = previous_date.strftime("%Y-%m-%d")

    try:
        idx_location = df.index.get_loc(prev_str)
    except KeyError:
        # Try nearest previous trading day
        try:
            idx_location = df.index.get_indexer([previous_date], method="pad")[0]
        except Exception:
            raise HTTPException(
                status_code=400,
                detail=f"previous_date {prev_str} not found in data or before available history",
            )

    if idx_location - feature_length < 0:
        raise HTTPException(
            status_code=400,
            detail="Not enough history before previous_date for given feature_length",
        )

    Features = df.iloc[idx_location - feature_length: idx_location, :].values
    Features = np.expand_dims(Features, axis=0)           # (1, time_steps, features)
    Features = Feature_Scaler.transform(Features)
    Prediction = model.predict(Features)
    Prediction = Target_Scaler.inverse_transform(Prediction)
    return float(Prediction[0][0])


def predict_multiple_days(model, df: pd.DataFrame, previous_date: datetime, days: int = 7, feature_length: int = 32):
    df_copy = df.copy()
    if not isinstance(df_copy.index, pd.DatetimeIndex):
        df_copy.index = pd.to_datetime(df_copy.index)

    preds = []
    dates = []

    current_date = previous_date

    for _ in range(days):
        pred = predict_one_day(model, df_copy, current_date, feature_length)
        next_date = current_date + timedelta(days=1)

        # Append prediction into df_copy for iterative forecasting
        new_row = df_copy.iloc[-1:].copy()
        new_row.index = [next_date]
        new_row["Close"] = pred
        df_copy = pd.concat([df_copy, new_row])

        preds.append(pred)
        dates.append(next_date)
        current_date = next_date

    return dates, preds


def load_stock_data(ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        raise HTTPException(status_code=400, detail="No data returned for given ticker/date range")
    return df


# ============================
# Routes
# ============================
@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": Model is not None}


@app.post("/predict/next-day", response_model=SinglePredictionResponse)
def predict_next_day(req: SinglePredictionRequest):
    df = load_stock_data(req.ticker, req.start_date, req.end_date)
    price = predict_one_day(Model, df, req.previous_date, req.feature_length)
    next_date = req.previous_date + timedelta(days=1)
    return SinglePredictionResponse(
        ticker=req.ticker,
        previous_date=req.previous_date,
        next_date=next_date,
        predicted_price=price,
    )


@app.post("/predict/multi-day", response_model=MultiDayPredictionResponse)
def predict_multi_day(req: MultiDayPredictionRequest):
    if req.days < 1:
        raise HTTPException(status_code=400, detail="days must be >= 1")

    df = load_stock_data(req.ticker, req.start_date, req.end_date)
    dates, preds = predict_multiple_days(
        Model, df, req.previous_date, days=req.days, feature_length=req.feature_length
    )

    forecast = [ForecastPoint(date=d, predicted_price=p) for d, p in zip(dates, preds)]

    return MultiDayPredictionResponse(
        ticker=req.ticker,
        start_date=req.start_date,
        previous_date=req.previous_date,
        days=req.days,
        forecast=forecast,
    )
