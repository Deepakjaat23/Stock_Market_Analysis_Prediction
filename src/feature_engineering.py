"""
feature_engineering.py
----------------------
Creates technical indicators for stock prediction.
"""

import os
import numpy as np
import pandas as pd

PROCESSED_FOLDER = "data/processed"


# -------------------------------------------------------
# Load CSV
# -------------------------------------------------------

def load_data(file_path):
    return pd.read_csv(file_path, parse_dates=["Date"])


# -------------------------------------------------------
# Create Features
# -------------------------------------------------------

def create_features(df):

    print("Creating Features...")

    # Daily Return
    df["Daily_Return"] = df["Close"].pct_change()

    # Log Return
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

    # Moving Averages
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA100"] = df["Close"].rolling(100).mean()

    # Exponential Moving Average
    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()

    # Volatility
    df["Volatility"] = df["Daily_Return"].rolling(20).std()

    # Dollar Volume
    df["Dollar_Volume"] = df["Close"] * df["Volume"]

    # Price Change %
    df["Price_Change_%"] = (
        (df["Close"] - df["Open"]) / df["Open"]
    ) * 100

    # Volume Change %
    df["Volume_Change_%"] = df["Volume"].pct_change() * 100

    # High Low Spread
    df["High_Low_Spread"] = df["High"] - df["Low"]

    # Open Close Change
    df["Open_Close_Change"] = df["Close"] - df["Open"]

    # Cumulative Return
    df["Cumulative_Return"] = (
        df["Close"] / df["Close"].iloc[0]
    ) - 1

    # Bollinger Bands
    rolling_std = df["Close"].rolling(20).std()

    df["BB_Upper"] = df["MA20"] + 2 * rolling_std

    df["BB_Lower"] = df["MA20"] - 2 * rolling_std

    # RSI (14)

    delta = df["Close"].diff()

    gain = delta.where(delta > 0, 0)

    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(14).mean()

    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD

    ema12 = df["Close"].ewm(span=12).mean()

    ema26 = df["Close"].ewm(span=26).mean()

    df["MACD"] = ema12 - ema26

    df["Signal_Line"] = df["MACD"].ewm(span=9).mean()

    # ATR

    high_low = df["High"] - df["Low"]

    high_close = abs(df["High"] - df["Close"].shift())

    low_close = abs(df["Low"] - df["Close"].shift())

    tr = pd.concat(
        [high_low, high_close, low_close],
        axis=1
    ).max(axis=1)

    df["ATR"] = tr.rolling(14).mean()

    # Remove rows with NaN due to rolling windows

    df.dropna(inplace=True)

    df.reset_index(drop=True, inplace=True)

    print("Feature Engineering Completed")

    return df


# -------------------------------------------------------
# Save CSV
# -------------------------------------------------------

def save_feature_data(df, company):

    path = os.path.join(PROCESSED_FOLDER, f"{company}.csv")

    df.to_csv(path, index=False)

    print(f"Updated -> {company}.csv")


# -------------------------------------------------------
# Process All CSV
# -------------------------------------------------------

def process_all():

    files = [
        f for f in os.listdir(PROCESSED_FOLDER)
        if f.endswith(".csv")
    ]

    datasets = {}

    for file in files:

        company = file.replace(".csv", "")

        print("\n--------------------------")
        print(company)

        path = os.path.join(PROCESSED_FOLDER, file)

        df = load_data(path)

        df = create_features(df)

        save_feature_data(df, company)

        datasets[company] = df

    return datasets


# -------------------------------------------------------
# Testing
# -------------------------------------------------------

if __name__ == "__main__":

    datasets = process_all()

    print("\nCompleted")

    print(list(datasets.keys()))