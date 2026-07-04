"""
model_utils.py
--------------
Core utilities used by prediction.py to:
    - load a processed (feature engineered) stock CSV
    - build X (features) / y (target) for supervised learning
    - split into train/test sets (time-ordered, no shuffle)
    - scale features (optional, kept for models that need it)
    - train a model, evaluate it, and save it to disk
    - compare several trained models and pick the best one
    - reload a saved model for later predictions

Every time the project is run, models are retrained on whatever data
is currently inside data/processed, so the "best model" naturally
refreshes itself based on the latest dataset.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

plt.style.use("ggplot")

#-----------------------------------------------------Folder Paths--------------------------------------------

MODELS_FOLDER = "models"
REPORTS_FOLDER = "reports"
GRAPHS_FOLDER = "graphs"


#-----------------------------------------------------Feature List--------------------------------------------
# Must match the columns produced by feature_engineering.py

FEATURE_COLUMNS = [

    "Open",
    "High",
    "Low",
    "Close",
    "Volume",

    "MA20",
    "MA50",
    "MA100",

    "EMA20",
    "EMA50",

    "Volatility",

    "Dollar_Volume",

    "Price_Change_%",
    "Volume_Change_%",
    "High_Low_Spread",
    "Open_Close_Change",

    "RSI",
    "MACD",
    "Signal_Line",
    "ATR"

]

TARGET_COLUMN = "Close"


#-----------------------------------------------------Load Dataset--------------------------------------------

def load_dataset(file_path):
    """
    Load a processed CSV (already feature engineered) and sort it by date.
    """

    df = pd.read_csv(file_path, parse_dates=["Date"])

    df.sort_values("Date", inplace=True)

    df.reset_index(drop=True, inplace=True)

    return df


#-----------------------------------------------------Prepare Data--------------------------------------------

def prepare_data(df):
    """
    Build features (X) and target (y).

    Target = next day's Close price, so the model learns to predict
    tomorrow's closing price from today's indicators.
    """

    data = df.copy()

    data["Target"] = data[TARGET_COLUMN].shift(-1)

    data.dropna(inplace=True)

    data.reset_index(drop=True, inplace=True)

    X = data[FEATURE_COLUMNS]

    y = data["Target"]

    return X, y


#-----------------------------------------------------Split Data--------------------------------------------

def split_data(X, y, test_size=0.2):
    """
    Time-ordered split (no shuffle) since this is time series data -
    the model is always tested on the most recent, unseen days.
    """

    split_index = int(len(X) * (1 - test_size))

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test


#-----------------------------------------------------Scale Data--------------------------------------------

def scale_data(X_train, X_test):
    """
    Standardize features. Kept available for models that benefit from
    scaling (e.g. Linear Regression); tree-based models don't need it.
    """

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    return scaler, X_train_scaled, X_test_scaled


#-----------------------------------------------------Train Model--------------------------------------------

def train_model(model, model_name, X_train, X_test, y_train, y_test, scale=False):
    """
    Fit a model, evaluate it on the test set, print the metrics,
    save it to disk, and return a results dictionary.
    """

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    mae = mean_absolute_error(y_test, predictions)

    r2 = r2_score(y_test, predictions)

    print(f"\n{model_name}")

    print(f"   RMSE : {rmse:.4f}")

    print(f"   MAE  : {mae:.4f}")

    print(f"   R2   : {r2:.4f}")

    save_model(model, model_name)

    return {
        "name": model_name,
        "model": model,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }


#-----------------------------------------------------Compare Models--------------------------------------------

def compare_models(results):
    """
    Rank trained models by RMSE (lower is better), print a comparison
    table, save it as a report, and return the name of the best model.
    """

    os.makedirs(REPORTS_FOLDER, exist_ok=True)

    comparison = pd.DataFrame(results)[["name", "rmse", "mae", "r2"]]

    comparison.sort_values("rmse", inplace=True)

    comparison.reset_index(drop=True, inplace=True)

    print("\n" + "=" * 55)

    print("MODEL COMPARISON (sorted by RMSE, lower RMSE is better)")

    print("=" * 55)

    print(comparison.to_string(index=False))

    report_path = os.path.join(REPORTS_FOLDER, "model_comparison.csv")

    comparison.to_csv(report_path, index=False)

    print(f"\nComparison report saved -> {report_path}")

    best_model_name = comparison.iloc[0]["name"]

    return best_model_name


#-----------------------------------------------------Plot Model Comparison--------------------------------------------

def plot_model_comparison(results, company="Stock"):
    """
    Draws a 3-panel bar chart (RMSE, MAE, R2) comparing every trained
    model, highlights the best model (lowest RMSE) in gold, and saves
    the figure into the graphs/ folder.
    """

    os.makedirs(GRAPHS_FOLDER, exist_ok=True)

    comparison = pd.DataFrame(results)[["name", "rmse", "mae", "r2"]]

    comparison.sort_values("rmse", inplace=True)

    comparison.reset_index(drop=True, inplace=True)

    best_name = comparison.iloc[0]["name"]

    bar_colors = [
        "gold" if name == best_name else "steelblue"
        for name in comparison["name"]
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    metrics = [
        ("rmse", "RMSE (Lower is Better)"),
        ("mae", "MAE (Lower is Better)"),
        ("r2", "R2 Score (Higher is Better)")
    ]

    for ax, (col, title) in zip(axes, metrics):

        bars = ax.bar(comparison["name"], comparison[col], color=bar_colors)

        ax.set_title(title, fontsize=13)

        ax.set_xticks(range(len(comparison["name"])))

        ax.set_xticklabels(comparison["name"], rotation=20, ha="right")

        for bar, value in zip(bars, comparison[col]):

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

    fig.suptitle(
        f"Model Comparison - {company}  (Best Model : {best_name})",
        fontsize=15
    )

    plt.tight_layout()

    file_path = os.path.join(
        GRAPHS_FOLDER,
        f"{company}_model_comparison.png"
    )

    plt.savefig(file_path, dpi=150)

    plt.close(fig)

    print(f"Comparison graph saved -> {file_path}")

    return file_path


#-----------------------------------------------------Best Model Summary--------------------------------------------

def save_best_model_summary(best_model_name, results, company="Stock"):
    """
    Writes a short text report naming the best model and its accuracy
    (R2 score expressed as a percentage), along with RMSE/MAE, into
    the reports/ folder.
    """

    os.makedirs(REPORTS_FOLDER, exist_ok=True)

    comparison = pd.DataFrame(results)

    best_row = comparison[comparison["name"] == best_model_name].iloc[0]

    accuracy = best_row["r2"] * 100

    lines = [
        "=" * 50,
        f"BEST MODEL REPORT - {company}",
        "=" * 50,
        f"Best Model      : {best_model_name}",
        f"Accuracy (R2)   : {accuracy:.2f}%",
        f"RMSE            : {best_row['rmse']:.4f}",
        f"MAE             : {best_row['mae']:.4f}",
        "=" * 50
    ]

    text = "\n".join(lines)

    print("\n" + text)

    file_path = os.path.join(
        REPORTS_FOLDER,
        f"{company}_best_model.txt"
    )

    with open(file_path, "w") as f:
        f.write(text)

    print(f"\nBest model summary saved -> {file_path}")

    return file_path


#-----------------------------------------------------Save Model--------------------------------------------

def save_model(model, model_name):
    """
    Save a trained model to the models/ folder as <model_name>.pkl
    """

    os.makedirs(MODELS_FOLDER, exist_ok=True)

    file_name = model_name.replace(" ", "_").lower() + ".pkl"

    file_path = os.path.join(MODELS_FOLDER, file_name)

    joblib.dump(model, file_path)

    print(f"Model saved -> {file_path}")


#-----------------------------------------------------Load Model--------------------------------------------

def load_model(model_file):
    """
    Load a previously saved model from the models/ folder.
    """

    file_path = os.path.join(MODELS_FOLDER, model_file)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")

    return joblib.load(file_path)
