"""
comparison.py
-------------
Multiple Stock Comparison Module

Loads several processed stock datasets side by side and compares them on
normalized price growth, returns, risk, and correlation - useful for
deciding between candidates rather than analyzing one stock in isolation.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROCESSED_FOLDER = "data/processed"
GRAPHS_FOLDER = "graphs"
REPORTS_FOLDER = "reports"

TRADING_DAYS = 252


# --------------------------------------------------------------------
# Data Loading
# --------------------------------------------------------------------

def load_companies(tickers):

    datasets = {}

    for ticker in tickers:

        path = os.path.join(PROCESSED_FOLDER, f"{ticker}.csv")

        if os.path.exists(path):

            datasets[ticker] = pd.read_csv(path, parse_dates=["Date"])

        else:

            print(f"Skipping {ticker}: processed file not found.")

    return datasets


# --------------------------------------------------------------------
# Normalized Growth (base 100)
# --------------------------------------------------------------------

def plot_normalized_growth(datasets, save=True):

    plt.figure(figsize=(14, 7))

    for company, df in datasets.items():

        normalized = df["Close"] / df["Close"].iloc[0] * 100

        plt.plot(df["Date"], normalized, linewidth=2, label=company)

    plt.title("Normalized Price Growth (Base = 100)")

    plt.xlabel("Date")

    plt.ylabel("Growth (Base 100)")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    if save:

        os.makedirs(GRAPHS_FOLDER, exist_ok=True)

        path = os.path.join(GRAPHS_FOLDER, "normalized_growth_comparison.png")

        plt.savefig(path, dpi=150)

        print(f"Saved -> {path}")

    plt.show()


# --------------------------------------------------------------------
# Comparison Table
# --------------------------------------------------------------------

def build_comparison_table(datasets):
    """
    One row per company: total return, annualized volatility, Sharpe,
    max drawdown, average RSI, current price.
    """

    rows = []

    for company, df in datasets.items():

        daily_return = df["Close"].pct_change().dropna()

        total_return = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100

        ann_return = daily_return.mean() * TRADING_DAYS * 100

        ann_vol = daily_return.std() * np.sqrt(TRADING_DAYS) * 100

        sharpe = (ann_return / ann_vol) if ann_vol != 0 else 0

        cumulative = (1 + daily_return).cumprod()

        running_max = cumulative.cummax()

        drawdown = ((cumulative - running_max) / running_max).min() * 100

        row = {
            "Company": company,
            "Current_Price": round(df["Close"].iloc[-1], 2),
            "Total_Return_%": round(total_return, 2),
            "Annual_Return_%": round(ann_return, 2),
            "Annual_Volatility_%": round(ann_vol, 2),
            "Sharpe_Ratio": round(sharpe, 3),
            "Max_Drawdown_%": round(drawdown, 2),
        }

        if "RSI" in df.columns:

            row["Latest_RSI"] = round(df["RSI"].iloc[-1], 2)

        rows.append(row)

    table = pd.DataFrame(rows)

    table.sort_values("Total_Return_%", ascending=False, inplace=True)

    table.reset_index(drop=True, inplace=True)

    return table


def plot_comparison_bar(table, save=True):

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].bar(table["Company"], table["Total_Return_%"], color="steelblue")

    axes[0].set_title("Total Return %")

    axes[0].axhline(0, color="black", linewidth=0.8)

    axes[1].bar(table["Company"], table["Sharpe_Ratio"], color="seagreen")

    axes[1].set_title("Sharpe Ratio")

    axes[1].axhline(0, color="black", linewidth=0.8)

    plt.tight_layout()

    if save:

        os.makedirs(GRAPHS_FOLDER, exist_ok=True)

        path = os.path.join(GRAPHS_FOLDER, "comparison_bar_chart.png")

        plt.savefig(path, dpi=150)

        print(f"Saved -> {path}")

    plt.show()


def save_comparison_report(table):

    os.makedirs(REPORTS_FOLDER, exist_ok=True)

    path = os.path.join(REPORTS_FOLDER, "stock_comparison.csv")

    table.to_csv(path, index=False)

    print(f"Saved -> {path}")


# --------------------------------------------------------------------
# CLI Menu
# --------------------------------------------------------------------

def _select_tickers():

    files = [f[:-4] for f in os.listdir(PROCESSED_FOLDER) if f.endswith(".csv")]

    if not files:

        print("No processed data found. Run Preprocessing first.")

        return []

    print("\nAvailable Companies:", ", ".join(files))

    raw = input("Enter tickers to compare (comma separated, blank = all): ").upper().strip()

    if not raw:

        return files

    return [t.strip() for t in raw.split(",") if t.strip() in files]


def comparison_menu():

    while True:

        print("\n" + "=" * 55)
        print("        MULTI-STOCK COMPARISON MENU")
        print("=" * 55)

        print("1. Normalized Growth Chart")
        print("2. Comparison Table (Return / Risk / Sharpe)")
        print("3. Comparison Bar Charts")
        print("0. Back")

        choice = input("\nEnter your choice : ")

        if choice == "1":

            tickers = _select_tickers()

            if len(tickers) < 2:

                print("Need at least 2 companies to compare.")

                continue

            datasets = load_companies(tickers)

            plot_normalized_growth(datasets)

        elif choice == "2":

            tickers = _select_tickers()

            if len(tickers) < 2:

                print("Need at least 2 companies to compare.")

                continue

            datasets = load_companies(tickers)

            table = build_comparison_table(datasets)

            print("\n" + table.to_string(index=False))

            save_comparison_report(table)

        elif choice == "3":

            tickers = _select_tickers()

            if len(tickers) < 2:

                print("Need at least 2 companies to compare.")

                continue

            datasets = load_companies(tickers)

            table = build_comparison_table(datasets)

            plot_comparison_bar(table)

        elif choice == "0":

            break

        else:

            print("Invalid Choice")


if __name__ == "__main__":

    comparison_menu()
