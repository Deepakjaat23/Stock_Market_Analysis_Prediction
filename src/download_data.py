"""
download_data.py
----------------
Downloads historical stock market data from Yahoo Finance
and saves it into the data/raw folder.
"""

import os
import pandas as pd
import yfinance as yf


# Folder to store raw datasets
RAW_DATA_FOLDER = "data/raw"


def get_stock_symbols():
    """
    Ask the user to enter one or more stock symbols.

    Example:
        AAPL
        AAPL,MSFT,TSLA
        RELIANCE.NS,TCS.NS
    """

    while True:

        symbols = input(
            "\nEnter Stock Symbol(s)\n"
            "(Example: AAPL, MSFT, TSLA or RELIANCE.NS): "
        ).upper().strip()

        tickers = [x.strip() for x in symbols.split(",") if x.strip()]

        if len(tickers) == 0:
            print("Please enter at least one stock symbol.\n")
        else:
            return tickers


def download_stock_data(ticker, period="5y"):
    """
    Download stock data for a single company.
    """

    try:

        print(f"Downloading {ticker}...")

        df = yf.Ticker(ticker).history(period=period)

        if df.empty:
            print(f"❌ No data found for {ticker}")
            return None

        df.reset_index(inplace=True)

        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

        df["Company"] = ticker

        return df

    except Exception as e:

        print(f"Error downloading {ticker}")

        print(e)

        return None


def save_stock_data(df, ticker):
    """
    Save dataframe as CSV.
    """

    os.makedirs(RAW_DATA_FOLDER, exist_ok=True)

    file_path = os.path.join(RAW_DATA_FOLDER, f"{ticker}.csv")

    df.to_csv(file_path, index=False)

    print(f"Saved -> {file_path}")


def download_multiple_stocks(tickers):
    """
    Download multiple stocks.

    Returns
    -------
    Dictionary

    {
        "AAPL": dataframe,
        "MSFT": dataframe
    }

    """

    stock_data = {}

    for ticker in tickers:

        df = download_stock_data(ticker)

        if df is not None:

            save_stock_data(df, ticker)

            stock_data[ticker] = df

    return stock_data


# --------------------------
# Testing
# --------------------------

if __name__ == "__main__":

    tickers = get_stock_symbols()

    stock_data = download_multiple_stocks(tickers)

    print("\nDownloaded Companies")

    print(list(stock_data.keys()))