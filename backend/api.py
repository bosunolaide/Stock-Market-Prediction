from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import math
import yfinance as yf

from stock_mlops.model_loader import load_model_bundle
from stock_mlops.logging_config import setup_logging

from datetime import datetime, timedelta

app = FastAPI(title="Stock Market Prediction API")

logger = setup_logging(__name__)
bundle = None

# ✅ Your saved Model.h5 expects input shape (None, 32, 3)
# These MUST match what the model/scalers were trained with.
# For legacy artifacts in this repo, the model expects 3 features.
REQUIRED_FEATURE_COLS = ["Open", "High", "Low"]




def _assert_finite(name: str, value):
    """Fail fast if model/scalers produce NaN/Inf (Starlette JSON refuses these)."""
    arr = np.asarray(value)
    if not np.all(np.isfinite(arr)):
        logger.error("Non-finite %s produced: %s", name, value)
        raise HTTPException(
            status_code=500,
            detail=f"Non-finite value produced for {name} (nan/inf). This usually means scaler/model mismatch or unstable multi-step forecasting.",
        )
    return value


def _safe_features(features: np.ndarray, feature_scaler: object) -> np.ndarray:
    """Harden model inputs so the API returns a stable response instead of crashing.

    - Replace NaN/Inf (often caused by missing yfinance bars or legacy scaler artifacts)
    - If the scaler looks like MinMaxScaler-based, clip to [0, 1]
    """
    x = np.asarray(features, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # Heuristic: MultiDimensionScaler(...scalers=[MinMaxScaler,...])
    try:
        scalers = getattr(feature_scaler, "scalers", None)
        if scalers and hasattr(scalers[0], "data_min_"):
            x = np.clip(x, 0.0, 1.0)
    except Exception:
        pass

    return x


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


def _ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df contains exactly the columns the model/scalers expect."""
    missing = [c for c in REQUIRED_FEATURE_COLS if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required columns from data: {missing}")
    return df[REQUIRED_FEATURE_COLS].copy()


def predict_one_day(model, df: pd.DataFrame, previous_date: datetime, feature_length: int = 32) -> float:
    if bundle is None:
        raise HTTPException(status_code=500, detail="Model bundle not loaded")

    feature_scaler = bundle.feature_scaler
    target_scaler = bundle.target_scaler

    if model is None or feature_scaler is None or target_scaler is None:
        raise HTTPException(status_code=500, detail="Model or scalers not loaded")

    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = _ensure_required_columns(df)

    ts = pd.Timestamp(previous_date)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)

    idx = df.index.get_indexer([ts], method="pad")[0]
    if idx == -1:
        raise HTTPException(
            status_code=400,
            detail=f"previous_date={ts.date()} is earlier than the first available date {df.index.min().date()}",
        )
    if idx < feature_length:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Not enough historical rows before {ts.date()} to build a window of length {feature_length}. "
                f"Need at least {feature_length} rows, but have {idx}."
            ),
        )

    features = df.iloc[idx - feature_length : idx, :].values
    features = np.expand_dims(features, axis=0)

    # Guard against feature/scaler mismatch
    n_feat = int(features.shape[2])
    n_scalers = len(getattr(feature_scaler, "scalers", []))
    if n_scalers and n_feat != n_scalers:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Feature mismatch: got {n_feat} features but feature_scaler has {n_scalers} scalers. "
                f"Ensure data columns match training: {REQUIRED_FEATURE_COLS}."
            ),
        )

    features = feature_scaler.transform(features)
    features = _safe_features(features, feature_scaler)

    prediction = model.predict(features)
    prediction = np.asarray(prediction).reshape(-1, 1)
    prediction = target_scaler.inverse_transform(prediction)
    # If model outputs NaN/Inf, we'll handle it at the multi-day loop level.
    if not np.all(np.isfinite(prediction)):
        logger.error("Non-finite prediction produced: %s", prediction)
        return float("nan")
    return float(prediction[0][0])


def predict_multiple_days(model, df: pd.DataFrame, previous_date: datetime, days: int = 7, feature_length: int = 32, last_close_like: float | None = None):
    df_copy = df.copy()
    df_copy.index = pd.to_datetime(df_copy.index)
    df_copy = df_copy.sort_index()
    df_copy = _ensure_required_columns(df_copy)

    preds, dates = [], []
    current_date = previous_date

    last_close = float(last_close_like) if last_close_like is not None else _get_last_close_like(df)

    for _ in range(days):
        pred_close = predict_one_day(model, df_copy, current_date, feature_length)
        if not np.isfinite(pred_close):
            # Graceful degradation: return a stable forecast instead of a 500.
            logger.warning(
                "Model produced non-finite prediction at %s; falling back to last_close=%s",
                current_date.date(),
                last_close,
            )
            pred_close = last_close
        last_close = float(pred_close)
        next_date = current_date + timedelta(days=1)

        # For multi-step forecasting, we must append a synthetic next-day row
        # with the SAME feature schema as training (3 features).
        # We set Open/High/Low to the predicted close as a simple, consistent proxy.
        new_row = pd.DataFrame(
            [[pred_close, pred_close, pred_close]],
            columns=REQUIRED_FEATURE_COLS,
            index=[pd.Timestamp(next_date)],
        )

        df_copy = pd.concat([df_copy, new_row])

        preds.append(float(pred_close))
        dates.append(next_date)
        current_date = next_date

    return dates, preds


@app.post("/predict/multi-day", response_model=MultiDayPredictionResponse)
def predict_multi_day(req: MultiDayPredictionRequest):
    if bundle is None or bundle.model is None:
        raise HTTPException(status_code=500, detail="Model bundle not loaded")

    df = yf.download(req.ticker, start=req.start_date, end=req.end_date)
    df = _normalize_yfinance_df(df, req.ticker)
    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="No data returned for ticker/date range")
    last_close_like = _get_last_close_like(df)
    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="No data returned for ticker/date range")

    # ✅ Enforce model training feature columns (3 features expected by Model.h5)
    df = _ensure_required_columns(df)

    dates, preds = predict_multiple_days(bundle.model, df, req.previous_date, req.days, req.feature_length, last_close_like=last_close_like)
    # Ensure JSON-safe output. If the model becomes unstable and produces NaN/Inf
    # we fall back to the last finite value instead of crashing the API.
    clean_preds = []
    last_finite = None
    for p in preds:
        if p is not None and np.isfinite(p):
            last_finite = float(p)
            clean_preds.append(float(p))
        else:
            # choose sensible fallback: last finite prediction or last known close
            fallback = last_finite
            if fallback is None:
                try:
                    fallback = float(df["Close"].iloc[-1])
                except Exception:
                    fallback = 0.0
            clean_preds.append(float(fallback))
    preds = clean_preds
    forecast = [ForecastPoint(date=d, predicted_price=p) for d, p in zip(dates, preds)]
    return MultiDayPredictionResponse(
        ticker=req.ticker,
        start_date=req.start_date,
        previous_date=req.previous_date,
        days=req.days,
        forecast=forecast,
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bundle is not None}


def _normalize_yfinance_df(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalize yfinance output to single-level OHLCV columns.

    yfinance can return:
      - single-level columns: ["Open","High","Low","Close","Adj Close","Volume"]
      - MultiIndex columns where either:
          (TICKER, field)  e.g. ("GOOGL","Open")
          (field, TICKER)  e.g. ("Open","GOOGL")
    We detect which level contains OHLCV field names and keep that level.
    """
    if df is None:
        return df
    out = df.copy()

    if isinstance(out.columns, pd.MultiIndex):
        level0 = [str(x) for x in out.columns.get_level_values(0)]
        level1 = [str(x) for x in out.columns.get_level_values(1)]
        known = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}

        s0 = len(set(level0) & known)
        s1 = len(set(level1) & known)

        if s0 > s1:
            # (field, ticker) -> keep fields
            out.columns = out.columns.get_level_values(0)
        elif s1 > s0:
            # (ticker, field) -> keep fields
            out.columns = out.columns.get_level_values(1)
        else:
            # Fallback: try last level
            out.columns = out.columns.get_level_values(-1)

            # If not unique, flatten fully
            if len(set(out.columns)) != len(out.columns):
                out.columns = ["|".join(map(str, tup)) for tup in out.columns.to_flat_index()]

    out.columns = [str(c) for c in out.columns]
    return out


def _get_last_close_like(df: pd.DataFrame) -> float:
    """Get a sensible fallback target value from raw yfinance frame."""
    for col in ("Close", "Adj Close", "Open", "High", "Low"):
        if col in df.columns:
            v = float(df[col].dropna().iloc[-1])
            if math.isfinite(v):
                return v
    # absolute last resort
    v = float(df.select_dtypes(include=["number"]).to_numpy().ravel()[-1])
    return v
