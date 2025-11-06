import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime
import os

st.set_page_config(page_title="Stock Market Prediction", page_icon="📈", layout="wide")
st.title("📈 Stock Market Prediction Dashboard")

API_URL = os.getenv("API_URL", "http://backend:8000")

try:
    r = requests.get(f"{API_URL}/health")
    if r.status_code == 200:
        st.sidebar.success("✅ Connected to FastAPI backend!")
    else:
        st.sidebar.warning("⚠️ FastAPI reachable but returned error.")
except Exception:
    st.sidebar.error("❌ Could not reach backend.")
    st.stop()

st.sidebar.header("Input Parameters")
ticker = st.sidebar.text_input("Stock Symbol (e.g. AAPL, TSLA, GOOGL)", "GOOGL")
start_date = st.sidebar.date_input("Start Date", datetime(2018, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.today())
previous_date = st.sidebar.text_input("Previous Date (YYYY-MM-DD)", "2021-01-14")
forecast_days = st.sidebar.slider("Days to Predict", 1, 30, 7)

if st.button("Predict Future Prices"):
    st.info("Requesting forecast from FastAPI...")
    payload = {
        "ticker": ticker,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "previous_date": previous_date,
        "days": forecast_days,
        "feature_length": 32
    }
    try:
        res = requests.post(f"{API_URL}/predict/multi-day", json=payload)
        if res.status_code != 200:
            st.error(res.text)
            st.stop()
        forecast = res.json()["forecast"]
        df = pd.DataFrame(forecast)
        df["date"] = pd.to_datetime(df["date"])
        fig = go.Figure(go.Scatter(x=df["date"], y=df["predicted_price"], mode="lines+markers", line=dict(color="orange")))
        fig.update_layout(title=f"{ticker} {forecast_days}-Day Forecast", xaxis_title="Date", yaxis_title="Price (USD)")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df)
    except Exception as e:
        st.error(f"❌ Request failed: {e}")
