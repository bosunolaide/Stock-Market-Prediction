import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime

# =========================================================
# 🎯 PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Stock Market Prediction", page_icon="📈", layout="wide")
st.title("📈 Stock Market Prediction Dashboard")
st.write("Frontend connected to a FastAPI backend inside the same container.")

# =========================================================
# ⚙️ API CONFIG
# =========================================================
API_URL = "http://localhost:8000"  # FastAPI runs on port 8000

# Health check
try:
    response = requests.get(f"{API_URL}/health")
    if response.status_code == 200:
        st.sidebar.success("✅ Connected to FastAPI backend!")
    else:
        st.sidebar.warning("⚠️ FastAPI backend reachable but returned an error.")
except Exception:
    st.sidebar.error("❌ Could not reach FastAPI backend.")
    st.stop()

# =========================================================
# 🧩 SIDEBAR INPUTS
# =========================================================
st.sidebar.header("Input Parameters")

ticker = st.sidebar.text_input("Stock Symbol (e.g. AAPL, TSLA, GOOGL)", "GOOGL")
start_date = st.sidebar.date_input("Start Date", datetime(2018, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.today())
previous_date = st.sidebar.text_input("Previous Date (YYYY-MM-DD)", "2021-01-14")
forecast_days = st.sidebar.slider("Days to Predict", 1, 30, 7)

# =========================================================
# 🚀 PREDICT USING FASTAPI
# =========================================================
if st.button("Predict Future Prices"):
    st.info("Sending request to FastAPI backend...")

    payload = {
        "ticker": ticker,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "previous_date": previous_date,
        "days": forecast_days,
        "feature_length": 32
    }

    try:
        response = requests.post(f"{API_URL}/predict/multi-day", json=payload)
        if response.status_code != 200:
            st.error(f"❌ API Error: {response.text}")
            st.stop()

        result = response.json()
        forecast = result["forecast"]

        # Convert forecast to DataFrame
        df_forecast = pd.DataFrame(forecast)
        df_forecast["date"] = pd.to_datetime(df_forecast["date"])
        df_forecast.rename(columns={"predicted_price": "Predicted"}, inplace=True)

        st.success(f"✅ Successfully predicted next {forecast_days} days!")

        # Plotly chart
        fig = go.Figure()

        # Historical line
        fig.add_trace(go.Scatter(
            x=[p["date"] for p in forecast],
            y=[p["predicted_price"] for p in forecast],
            mode='lines+markers',
            name='Predicted',
            line=dict(color='orange', width=3, dash='dot'),
            marker=dict(size=6, color='orange')
        ))

        # Chart layout
        fig.update_layout(
            title=f"{ticker} {forecast_days}-Day Forecast",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            hovermode="x unified",
            template="plotly_white",
            font=dict(size=14),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=60, b=40)
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📈 Forecast Data")
        st.dataframe(df_forecast)

        csv = df_forecast.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Forecast", csv, "forecast.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Failed to contact FastAPI: {e}")
