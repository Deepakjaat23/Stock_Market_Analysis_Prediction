"""
deep_learning.py
----------------
Deep Learning Prediction Module

Two independent forecasting approaches, kept separate because they solve
the problem differently:

    - LSTM   : a sequence model trained on a sliding window of past
               closing prices, good at capturing short-term patterns.
    - Prophet: Facebook's additive time-series model, good at capturing
               trend + seasonality with very little tuning.

Both are trained per-company on demand and save their artifacts to
models/lstm/ and models/prophet/ respectively.
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

PROCESSED_FOLDER = "data/processed"
GRAPHS_FOLDER = "graphs"
REPORTS_FOLDER = "reports"
LSTM_MODELS_FOLDER = "models/lstm"
PROPHET_MODELS_FOLDER = "models/prophet"

LOOKBACK = 60  # trading days used as the LSTM's input window


# ======================================================================
# LSTM
# ======================================================================

def _build_lstm_sequences(scaled_close, lookback=LOOKBACK):

    X, y = [], []

    for i in range(lookback, len(scaled_close)):

        X.append(scaled_close[i - lookback:i, 0])

        y.append(scaled_close[i, 0])

    X, y = np.array(X), np.array(y)

    X = X.reshape((X.shape[0], X.shape[1], 1))

    return X, y


def train_lstm_model(company, epochs=20, batch_size=32, lookback=LOOKBACK):
    """
    Trains an LSTM on a company's closing price history and saves the
    model + scaler to disk. Returns the trained model, scaler, and the
    full close-price series used (needed later to seed forecasts).
    """

    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    path = os.path.join(PROCESSED_FOLDER, f"{company}.csv")

    df = pd.read_csv(path, parse_dates=["Date"])

    df.sort_values("Date", inplace=True)

    close_prices = df[["Close"]].values

    if len(close_prices) <= lookback + 10:

        print(f"Not enough data to train an LSTM for {company} "
              f"(need > {lookback + 10} rows).")

        return None, None, None

    scaler = MinMaxScaler(feature_range=(0, 1))

    scaled = scaler.fit_transform(close_prices)

    # Time-ordered train/test split (last 10% held out)
    split = int(len(scaled) * 0.9)

    train_data = scaled[:split]

    test_data = scaled[split - lookback:]

    X_train, y_train = _build_lstm_sequences(train_data, lookback)

    X_test, y_test = _build_lstm_sequences(test_data, lookback)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mean_squared_error")

    print(f"\nTraining LSTM for {company} ({epochs} epochs)...")

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test) if len(X_test) > 0 else None,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # Evaluate
    if len(X_test) > 0:

        predictions = model.predict(X_test, verbose=0)

        predictions_actual = scaler.inverse_transform(predictions)

        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

        rmse = float(np.sqrt(np.mean((predictions_actual - y_test_actual) ** 2)))

        print(f"LSTM Test RMSE for {company} : {rmse:.4f}")

    os.makedirs(LSTM_MODELS_FOLDER, exist_ok=True)

    model_path = os.path.join(LSTM_MODELS_FOLDER, f"{company}_lstm.keras")

    scaler_path = os.path.join(LSTM_MODELS_FOLDER, f"{company}_scaler.pkl")

    model.save(model_path)

    joblib.dump(scaler, scaler_path)

    print(f"Saved -> {model_path}")

    print(f"Saved -> {scaler_path}")

    return model, scaler, close_prices


def load_lstm_model(company):

    from tensorflow.keras.models import load_model

    model_path = os.path.join(LSTM_MODELS_FOLDER, f"{company}_lstm.keras")

    scaler_path = os.path.join(LSTM_MODELS_FOLDER, f"{company}_scaler.pkl")

    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):

        return None, None

    model = load_model(model_path)

    scaler = joblib.load(scaler_path)

    return model, scaler


def predict_lstm_future(model, scaler, close_prices, days=30, lookback=LOOKBACK):
    """
    Iteratively forecasts `days` future closing prices by feeding each
    prediction back in as the newest point in the rolling window.
    """

    scaled = scaler.transform(close_prices)

    window = scaled[-lookback:].reshape(1, lookback, 1)

    predictions = []

    for _ in range(days):

        pred = model.predict(window, verbose=0)[0, 0]

        predictions.append(pred)

        window = np.append(window[:, 1:, :], [[[pred]]], axis=1)

    predictions = scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    ).flatten()

    return predictions


def plot_lstm_forecast(company, df, predictions, save=True):

    last_date = df["Date"].max()

    future_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1),
        periods=len(predictions)
    )

    plt.figure(figsize=(14, 7))

    plt.plot(df["Date"].tail(150), df["Close"].tail(150), label="Historical Close")

    plt.plot(future_dates, predictions, label="LSTM Forecast", linestyle="--", color="red")

    plt.title(f"{company} - LSTM Forecast ({len(predictions)} Days)")

    plt.xlabel("Date")

    plt.ylabel("Price")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    if save:

        os.makedirs(GRAPHS_FOLDER, exist_ok=True)

        path = os.path.join(GRAPHS_FOLDER, f"{company}_lstm_forecast.png")

        plt.savefig(path, dpi=150)

        print(f"Saved -> {path}")

    plt.show()


# ======================================================================
# Prophet
# ======================================================================

def train_prophet_model(company):
    """
    Trains a Prophet model on a company's Date/Close history and saves it.
    """

    from prophet import Prophet

    path = os.path.join(PROCESSED_FOLDER, f"{company}.csv")

    df = pd.read_csv(path, parse_dates=["Date"])

    prophet_df = df[["Date", "Close"]].rename(
        columns={"Date": "ds", "Close": "y"}
    )

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05
    )

    print(f"\nTraining Prophet model for {company}...")

    model.fit(prophet_df)

    os.makedirs(PROPHET_MODELS_FOLDER, exist_ok=True)

    model_path = os.path.join(PROPHET_MODELS_FOLDER, f"{company}_prophet.pkl")

    joblib.dump(model, model_path)

    print(f"Saved -> {model_path}")

    return model


def load_prophet_model(company):

    model_path = os.path.join(PROPHET_MODELS_FOLDER, f"{company}_prophet.pkl")

    if not os.path.exists(model_path):

        return None

    return joblib.load(model_path)


def predict_prophet_future(model, days=30):
    """
    Returns a DataFrame with columns: ds, yhat, yhat_lower, yhat_upper
    for the next `days` business days.
    """

    future = model.make_future_dataframe(periods=days, freq="B")

    forecast = model.predict(future)

    return forecast.tail(days)[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def plot_prophet_forecast(company, df, forecast, save=True):

    plt.figure(figsize=(14, 7))

    plt.plot(df["Date"].tail(150), df["Close"].tail(150), label="Historical Close")

    plt.plot(forecast["ds"], forecast["yhat"], label="Prophet Forecast", linestyle="--", color="green")

    plt.fill_between(
        forecast["ds"],
        forecast["yhat_lower"],
        forecast["yhat_upper"],
        alpha=0.2,
        color="green",
        label="Confidence Interval"
    )

    plt.title(f"{company} - Prophet Forecast ({len(forecast)} Days)")

    plt.xlabel("Date")

    plt.ylabel("Price")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    if save:

        os.makedirs(GRAPHS_FOLDER, exist_ok=True)

        path = os.path.join(GRAPHS_FOLDER, f"{company}_prophet_forecast.png")

        plt.savefig(path, dpi=150)

        print(f"Saved -> {path}")

    plt.show()


# ======================================================================
# CLI Menu
# ======================================================================

def _select_company():

    files = [f[:-4] for f in os.listdir(PROCESSED_FOLDER) if f.endswith(".csv")]

    if not files:

        print("No processed data found. Run Preprocessing first.")

        return None

    print("\nAvailable Companies\n")

    for i, f in enumerate(files, start=1):

        print(f"{i}. {f}")

    while True:

        try:

            choice = int(input("\nSelect Company : "))

            if 1 <= choice <= len(files):

                return files[choice - 1]

            print("Invalid Choice")

        except ValueError:

            print("Enter a valid number")


def deep_learning_menu():

    while True:

        print("\n" + "=" * 55)
        print("        DEEP LEARNING PREDICTION MENU")
        print("=" * 55)

        print("1. Train & Forecast with LSTM")
        print("2. Train & Forecast with Prophet")
        print("3. Compare LSTM vs Prophet")
        print("0. Back")

        choice = input("\nEnter your choice : ")

        if choice == "1":

            company = _select_company()

            if not company:
                continue

            days = int(input("Forecast how many days ahead? (e.g. 30): ") or 30)

            model, scaler, close_prices = train_lstm_model(company)

            if model is None:
                continue

            predictions = predict_lstm_future(model, scaler, close_prices, days=days)

            df = pd.read_csv(
                os.path.join(PROCESSED_FOLDER, f"{company}.csv"),
                parse_dates=["Date"]
            )

            print(f"\n{company} LSTM Forecast (next {days} days):")

            for i, p in enumerate(predictions, start=1):
                print(f"Day {i}: {p:.2f}")

            plot_lstm_forecast(company, df, predictions)

        elif choice == "2":

            company = _select_company()

            if not company:
                continue

            days = int(input("Forecast how many days ahead? (e.g. 30): ") or 30)

            model = train_prophet_model(company)

            forecast = predict_prophet_future(model, days=days)

            df = pd.read_csv(
                os.path.join(PROCESSED_FOLDER, f"{company}.csv"),
                parse_dates=["Date"]
            )

            print(f"\n{company} Prophet Forecast (next {days} days):")

            print(forecast.to_string(index=False))

            plot_prophet_forecast(company, df, forecast)

        elif choice == "3":

            company = _select_company()

            if not company:
                continue

            days = int(input("Forecast how many days ahead? (e.g. 30): ") or 30)

            df = pd.read_csv(
                os.path.join(PROCESSED_FOLDER, f"{company}.csv"),
                parse_dates=["Date"]
            )

            lstm_model, scaler, close_prices = train_lstm_model(company)

            lstm_preds = None

            if lstm_model is not None:

                lstm_preds = predict_lstm_future(lstm_model, scaler, close_prices, days=days)

            prophet_model = train_prophet_model(company)

            prophet_forecast = predict_prophet_future(prophet_model, days=days)

            plt.figure(figsize=(14, 7))

            plt.plot(df["Date"].tail(150), df["Close"].tail(150), label="Historical Close")

            if lstm_preds is not None:

                future_dates = pd.bdate_range(
                    start=df["Date"].max() + pd.Timedelta(days=1),
                    periods=len(lstm_preds)
                )

                plt.plot(future_dates, lstm_preds, linestyle="--", label="LSTM Forecast")

            plt.plot(
                prophet_forecast["ds"],
                prophet_forecast["yhat"],
                linestyle="--",
                label="Prophet Forecast"
            )

            plt.title(f"{company} - LSTM vs Prophet ({days} Days)")

            plt.legend()

            plt.grid(True)

            plt.tight_layout()

            os.makedirs(GRAPHS_FOLDER, exist_ok=True)

            path = os.path.join(GRAPHS_FOLDER, f"{company}_lstm_vs_prophet.png")

            plt.savefig(path, dpi=150)

            print(f"Saved -> {path}")

            plt.show()

        elif choice == "0":

            break

        else:

            print("Invalid Choice")


if __name__ == "__main__":

    deep_learning_menu()
