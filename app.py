import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# =========================================================
# 🎯 PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Stock Market Prediction", page_icon="📈", layout="wide")
st.title("📈 Stock Market Prediction Dashboard")
st.write("Predict stock prices using your trained deep learning model.")

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

@st.cache_data
def load_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    return df

data = load_data(ticker, start_date, end_date)
st.subheader(f"📅 Historical Data for {ticker}")
st.dataframe(data.tail())

# =========================================================
# 🔮 PREDICTION FUNCTION (from your notebook)
# =========================================================
def PredictStockPrice(Model, DataFrame, PreviousDate, feature_length=32):
    idx_location = DataFrame.index.get_loc(PreviousDate)
    Features = DataFrame.iloc[idx_location - feature_length: idx_location, :].values
    Features = np.expand_dims(Features, axis=0)
    Features = Feature_Scaler.transform(Features)
    Prediction = Model.predict(Features)
    Prediction = Target_Scaler.inverse_transform(Prediction)
    return Prediction[0][0]

# =========================================================
# 🚀 RUN PREDICTION
# =========================================================
if st.button("Predict Next Day Price"):
    try:
        prediction = PredictStockPrice(loaded_model, data, previous_date)
        st.success(f"💰 Predicted Closing Price after {previous_date}: **${prediction:.2f}**")

        # Plot last few days + predicted price
        last_dates = data.index[-32:]
        last_prices = data['Close'].iloc[-32:]
        future_date = pd.to_datetime(previous_date) + timedelta(days=1)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(last_dates, last_prices, label="Historical Close", color='blue')
        ax.scatter(future_date, prediction, color='orange', label="Predicted Price")
        ax.legend()
        ax.set_title(f"{ticker} Price Prediction")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# =========================================================
# 📤 EXPORT OPTION
# =========================================================
csv = data.to_csv().encode('utf-8')
st.download_button("📥 Download Historical Data", csv, "historical_data.csv", "text/csv")