import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go

# =========================================================
# 🎯 PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Stock Market Prediction", page_icon="📈", layout="wide")
st.title("📈 Stock Market Prediction Dashboard")
st.write("Interactively predict future stock prices using your trained deep learning model.")

# =========================================================
# 🧩 LOAD MODEL AND SCALERS
# =========================================================
@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model("Model.h5")
    with open("Feature_Scaler.pck", "rb") as f:
        feature_scaler = pickle.load(f)
    with open("Target_Scaler.pck", "rb") as f:
        target_scaler = pickle.load(f)
    return model, feature_scaler, target_scaler

try:
    loaded_model, Feature_Scaler, Target_Scaler = load_resources()
    st.sidebar.success("✅ Model and scalers loaded successfully!")
except Exception as e:
    st.sidebar.error(f"❌ Could not load model/scalers: {e}")
    st.stop()

# =========================================================
# 📊 LOAD HISTORICAL STOCK DATA
# =========================================================
st.sidebar.header("Input Parameters")

ticker = st.sidebar.text_input("Stock Symbol (e.g. AAPL, TSLA, GOOGL)", "GOOGL")
start_date = st.sidebar.date_input("Start Date", datetime(2018, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.today())
previous_date = st.sidebar.text_input("Previous Date (YYYY-MM-DD)", "2021-01-14")
forecast_days = st.sidebar.slider("Days to Predict", 1, 30, 7)

@st.cache_data
def load_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    return df

data = load_data(ticker, start_date, end_date)
st.subheader(f"📅 Historical Data for {ticker}")
st.dataframe(data.tail())

# =========================================================
# 🔮 PREDICTION FUNCTIONS
# =========================================================
def PredictStockPrice(Model, DataFrame, PreviousDate, feature_length=32):
    idx_location = DataFrame.index.get_loc(PreviousDate)
    Features = DataFrame.iloc[idx_location - feature_length: idx_location, :].values
    Features = np.expand_dims(Features, axis=0)
    Features = Feature_Scaler.transform(Features)
    Prediction = Model.predict(Features)
    Prediction = Target_Scaler.inverse_transform(Prediction)
    return Prediction[0][0]

def PredictMultipleDays(Model, DataFrame, start_date, days=7, feature_length=32):
    df_copy = DataFrame.copy()
    preds = []
    dates = []
    current_date = pd.to_datetime(start_date)

    for _ in range(days):
        try:
            pred = PredictStockPrice(Model, df_copy, current_date.strftime("%Y-%m-%d"), feature_length)
        except Exception:
            idx_location = df_copy.index.get_loc(df_copy.index[-1])
            Features = df_copy.iloc[idx_location - feature_length: idx_location, :].values
            Features = np.expand_dims(Features, axis=0)
            Features = Feature_Scaler.transform(Features)
            pred = Model.predict(Features)
            pred = Target_Scaler.inverse_transform(pred)[0][0]

        preds.append(pred)
        current_date += timedelta(days=1)
        dates.append(current_date)

        new_row = df_copy.iloc[-1:].copy()
        new_row.index = [current_date]
        new_row["Close"] = pred
        df_copy = pd.concat([df_copy, new_row])

    return pd.DataFrame({"Date": dates, "Predicted": preds})

# =========================================================
# 🚀 RUN PREDICTION
# =========================================================
if st.button("Predict Future Prices"):
    try:
        forecast_df = PredictMultipleDays(loaded_model, data, previous_date, forecast_days)
        st.success(f"✅ Successfully predicted the next {forecast_days} days!")

        # Plotly chart setup
        fig = go.Figure()

        # Historical prices
        fig.add_trace(go.Scatter(
            x=data.index[-90:],
            y=data["Close"].iloc[-90:],
            mode='lines',
            name='Historical',
            line=dict(color='royalblue', width=2)
        ))

        # Forecast prices
        fig.add_trace(go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df["Predicted"],
            mode='lines+markers',
            name='Forecast',
            line=dict(color='orange', width=3, dash='dot'),
            marker=dict(size=6, color='orange')
        ))

        # Shaded region for forecast period
        fig.add_vrect(
            x0=forecast_df["Date"].iloc[0],
            x1=forecast_df["Date"].iloc[-1],
            fillcolor="orange",
            opacity=0.1,
            line_width=0,
            annotation_text=f"{forecast_days}-Day Forecast",
            annotation_position="top left"
        )

        # Chart formatting
        fig.update_layout(
            title=f"{ticker} {forecast_days}-Day Price Forecast",
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
        st.dataframe(forecast_df)

        csv = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Forecast", csv, "forecast.csv", "text/csv")
    except Exception as e:
        st.error(f"❌ Forecast failed: {e}")
