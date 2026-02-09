import pickle
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

from stock_mlops.scalers import MultiDimensionScaler
from stock_mlops.settings import settings

FEATURE_LEN = 32
SYMBOL = "GOOGL"
START = "2018-01-01"
END = "2021-01-14"

def main():
    df = yf.download(SYMBOL, start=START, end=END, progress=False)
    df = df[["Close", "Volume"]].dropna()

    X, y = [], []
    for i in range(len(df) - FEATURE_LEN):
        X.append(df.iloc[i:i+FEATURE_LEN].values)
        y.append(df["Close"].iloc[i+FEATURE_LEN])

    X = np.array(X)                    # (n, 32, 2)
    y = np.array(y).reshape(-1, 1)      # (n, 1)

    feature_scaler = MultiDimensionScaler()
    feature_scaler.fit(X)     # notebook-style

    target_scaler = MinMaxScaler()
    target_scaler.fit(y)

    # Overwrite exactly the fallback paths used by the backend
    with open(settings.fallback_feature_scaler_path, "wb") as f:
        pickle.dump(feature_scaler, f)
    with open(settings.fallback_target_scaler_path, "wb") as f:
        pickle.dump(target_scaler, f)

    print("✅ Rebuilt fitted scalers at:")
    print(" -", settings.fallback_feature_scaler_path)
    print(" -", settings.fallback_target_scaler_path)

if __name__ == "__main__":
    main()
