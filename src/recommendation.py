"""
recommendation.py
------------------
Buy / Sell / Hold Recommendation Engine

Combines three independent signals into one weighted score:

    1. ML Prediction   (40%) - % change predicted by the best regression
                                model (falls back to LSTM if available)
    2. Technical Score (35%) - RSI, MACD, Bollinger Bands, moving average
                                trend, all rolled into a single -1..+1 score
    3. News Sentiment  (25%) - average VADER compound score of recent
                                headlines about the company

Final weighted score is mapped to a STRONG BUY / BUY / HOLD / SELL /
STRONG SELL label. Weights are deliberately explicit and adjustable so
the reasoning stays transparent rather than a black box.
"""

import os
import pandas as pd

from src.model_utils import FEATURE_COLUMNS, load_dataset, load_model
from src import sentiment as sentiment_module

PROCESSED_FOLDER = "data/processed"
REPORTS_FOLDER = "reports"

WEIGHT_ML = 0.40
WEIGHT_TECHNICAL = 0.35
WEIGHT_SENTIMENT = 0.25


# --------------------------------------------------------------------
# 1. ML Signal
# --------------------------------------------------------------------

def get_ml_signal(company, best_model_name):
    """
    Uses an already-trained regression model (see src/prediction.py) to
    predict tomorrow's close and converts the expected % change into a
    score in [-1, 1].

    Returns (score, details_dict) or (0.0, None) if no model is available.
    """

    model_file = best_model_name.replace(" ", "_").lower() + ".pkl"

    try:

        model = load_model(model_file)

    except FileNotFoundError:

        return 0.0, None

    path = os.path.join(PROCESSED_FOLDER, f"{company}.csv")

    df = load_dataset(path)

    latest = df.iloc[-1:]

    features = latest[FEATURE_COLUMNS]

    prediction = model.predict(features)[0]

    current_price = latest["Close"].values[0]

    percent_change = ((prediction - current_price) / current_price) * 100

    # Clip so a single extreme prediction can't swamp the blended score
    score = max(-1.0, min(1.0, percent_change / 5.0))

    details = {
        "model": best_model_name,
        "current_price": round(float(current_price), 2),
        "predicted_price": round(float(prediction), 2),
        "expected_change_%": round(float(percent_change), 2)
    }

    return score, details


# --------------------------------------------------------------------
# 2. Technical Signal
# --------------------------------------------------------------------

def get_technical_signal(company):
    """
    Combines RSI, MACD, Bollinger Bands, and moving-average trend into
    a single score in [-1, 1]. Positive = bullish, negative = bearish.
    """

    path = os.path.join(PROCESSED_FOLDER, f"{company}.csv")

    df = pd.read_csv(path, parse_dates=["Date"])

    latest = df.iloc[-1]

    sub_scores = []

    details = {}

    # --- RSI: oversold (<30) bullish, overbought (>70) bearish ---
    rsi = latest.get("RSI")

    if pd.notna(rsi):

        if rsi < 30:
            rsi_score = 1.0
        elif rsi > 70:
            rsi_score = -1.0
        else:
            rsi_score = (50 - rsi) / 20  # gentle slope around neutral 50

        rsi_score = max(-1.0, min(1.0, rsi_score))

        sub_scores.append(rsi_score)

        details["RSI"] = round(float(rsi), 2)

    # --- MACD: MACD above signal line = bullish ---
    macd = latest.get("MACD")

    signal_line = latest.get("Signal_Line")

    if pd.notna(macd) and pd.notna(signal_line):

        macd_score = 1.0 if macd > signal_line else -1.0

        sub_scores.append(macd_score)

        details["MACD"] = round(float(macd), 3)

        details["Signal_Line"] = round(float(signal_line), 3)

    # --- Bollinger Bands: price near lower band = bullish, near upper = bearish ---
    close = latest.get("Close")

    bb_upper = latest.get("BB_Upper")

    bb_lower = latest.get("BB_Lower")

    if pd.notna(close) and pd.notna(bb_upper) and pd.notna(bb_lower) and (bb_upper != bb_lower):

        position = (close - bb_lower) / (bb_upper - bb_lower)  # 0 = at lower band, 1 = at upper band

        bb_score = 1 - 2 * position  # 0 -> +1 (bullish), 1 -> -1 (bearish)

        bb_score = max(-1.0, min(1.0, bb_score))

        sub_scores.append(bb_score)

        details["BB_Position_%"] = round(float(position * 100), 1)

    # --- Moving Average Trend: price above MA20/MA50 = bullish ---
    ma20 = latest.get("MA20")

    ma50 = latest.get("MA50")

    if pd.notna(close) and pd.notna(ma20) and pd.notna(ma50):

        above_ma20 = 1 if close > ma20 else -1

        above_ma50 = 1 if close > ma50 else -1

        ma_score = (above_ma20 + above_ma50) / 2

        sub_scores.append(ma_score)

        details["Above_MA20"] = bool(close > ma20)

        details["Above_MA50"] = bool(close > ma50)

    if not sub_scores:

        return 0.0, details

    overall = sum(sub_scores) / len(sub_scores)

    return overall, details


# --------------------------------------------------------------------
# 3. Sentiment Signal
# --------------------------------------------------------------------

def get_sentiment_signal(company):
    """
    Fetches recent headlines and returns the average VADER score
    (already in roughly [-1, 1]) plus a details dict.
    """

    df, summary = sentiment_module.get_stock_sentiment(company, limit=15)

    return summary["average_score"], summary


# --------------------------------------------------------------------
# Combine
# --------------------------------------------------------------------

def classify_final_score(score):

    if score >= 0.5:
        return "STRONG BUY"

    elif score >= 0.15:
        return "BUY"

    elif score <= -0.5:
        return "STRONG SELL"

    elif score <= -0.15:
        return "SELL"

    else:
        return "HOLD"


def generate_recommendation(company, best_model_name=None):
    """
    Runs all three signals and blends them into one recommendation.

    If `best_model_name` isn't supplied (no trained regression model yet),
    the ML component is skipped and its weight is redistributed across
    the remaining signals.
    """

    print(f"\nGenerating recommendation for {company}...")

    weights = {
        "ml": WEIGHT_ML,
        "technical": WEIGHT_TECHNICAL,
        "sentiment": WEIGHT_SENTIMENT
    }

    ml_score, ml_details = (0.0, None)

    if best_model_name:

        ml_score, ml_details = get_ml_signal(company, best_model_name)

    if ml_details is None:

        # Redistribute the ML weight proportionally to the other two
        weights["ml"] = 0.0

        total_remaining = weights["technical"] + weights["sentiment"]

        weights["technical"] = WEIGHT_TECHNICAL / total_remaining

        weights["sentiment"] = WEIGHT_SENTIMENT / total_remaining

    technical_score, technical_details = get_technical_signal(company)

    sentiment_score, sentiment_details = get_sentiment_signal(company)

    final_score = (
        ml_score * weights["ml"] +
        technical_score * weights["technical"] +
        sentiment_score * weights["sentiment"]
    )

    label = classify_final_score(final_score)

    result = {
        "company": company,
        "recommendation": label,
        "final_score": round(final_score, 4),
        "weights_used": weights,
        "ml_signal": {"score": round(ml_score, 4), "details": ml_details},
        "technical_signal": {"score": round(technical_score, 4), "details": technical_details},
        "sentiment_signal": {"score": round(sentiment_score, 4), "details": sentiment_details},
    }

    return result


def print_recommendation(result):

    print("\n" + "=" * 55)

    print(f"RECOMMENDATION - {result['company']}")

    print("=" * 55)

    print(f"FINAL CALL : {result['recommendation']}  (score: {result['final_score']})")

    print("\n--- ML Prediction Signal ---")

    if result["ml_signal"]["details"]:

        for k, v in result["ml_signal"]["details"].items():
            print(f"  {k}: {v}")

    else:

        print("  Not available (train a model first via the Prediction menu)")

    print("\n--- Technical Signal ---")

    for k, v in (result["technical_signal"]["details"] or {}).items():
        print(f"  {k}: {v}")

    print("\n--- Sentiment Signal ---")

    print(f"  Label: {result['sentiment_signal']['details']['label']}")

    print(f"  Average Score: {result['sentiment_signal']['details']['average_score']}")

    print(f"  Headlines Analyzed: {result['sentiment_signal']['details']['total']}")

    print("=" * 55)


def save_recommendation_report(result):

    os.makedirs(REPORTS_FOLDER, exist_ok=True)

    path = os.path.join(REPORTS_FOLDER, f"{result['company']}_recommendation.txt")

    lines = [
        "=" * 50,
        f"RECOMMENDATION REPORT - {result['company']}",
        "=" * 50,
        f"Final Call   : {result['recommendation']}",
        f"Final Score  : {result['final_score']}",
        f"Weights Used : {result['weights_used']}",
        "",
        f"ML Signal        : {result['ml_signal']}",
        f"Technical Signal  : {result['technical_signal']}",
        f"Sentiment Signal  : {result['sentiment_signal']}",
        "=" * 50
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines))

    print(f"\nSaved -> {path}")


# --------------------------------------------------------------------
# CLI Menu
# --------------------------------------------------------------------

def recommendation_menu():

    while True:

        print("\n" + "=" * 55)
        print("        BUY / SELL RECOMMENDATION MENU")
        print("=" * 55)

        print("1. Generate Recommendation for a Company")
        print("0. Back")

        choice = input("\nEnter your choice : ")

        if choice == "1":

            files = [f[:-4] for f in os.listdir(PROCESSED_FOLDER) if f.endswith(".csv")]

            if not files:

                print("No processed data found. Run Preprocessing first.")

                continue

            print("\nAvailable Companies:", ", ".join(files))

            company = input("Enter company/ticker: ").upper().strip()

            if company not in files:

                print("Company not found in processed data.")

                continue

            model_name = input(
                "Best model name if already trained (e.g. 'XGBoost'), "
                "or leave blank to skip ML signal: "
            ).strip()

            result = generate_recommendation(company, best_model_name=model_name or None)

            print_recommendation(result)

            save_recommendation_report(result)

        elif choice == "0":

            break

        else:

            print("Invalid Choice")


if __name__ == "__main__":

    recommendation_menu()
