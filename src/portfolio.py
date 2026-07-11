"""
portfolio.py
------------
Portfolio Analysis Module

Combines multiple processed stock datasets into a portfolio and computes:
    - Individual & portfolio daily/cumulative returns
    - Annualized return, volatility, Sharpe ratio, max drawdown
    - Correlation matrix between holdings
    - Monte-Carlo random portfolio simulation to estimate the
      Efficient Frontier and the max-Sharpe / min-volatility portfolios
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROCESSED_FOLDER = "data/processed"
REPORTS_FOLDER = "reports"
GRAPHS_FOLDER = "graphs"

TRADING_DAYS = 252
RISK_FREE_RATE = 0.05  # annualized, adjust as needed


# --------------------------------------------------------------------
# Data Loading
# --------------------------------------------------------------------

def load_close_prices(tickers):
    """
    Load the 'Close' column for each ticker from data/processed and
    align them on Date into a single wide DataFrame.
    """

    frames = []

    for ticker in tickers:

        path = os.path.join(PROCESSED_FOLDER, f"{ticker}.csv")

        if not os.path.exists(path):

            print(f"Skipping {ticker}: processed file not found.")

            continue

        df = pd.read_csv(path, parse_dates=["Date"])[["Date", "Close"]]

        df.rename(columns={"Close": ticker}, inplace=True)

        df.set_index("Date", inplace=True)

        frames.append(df)

    if not frames:

        return pd.DataFrame()

    combined = pd.concat(frames, axis=1, join="inner")

    combined.sort_index(inplace=True)

    return combined


# --------------------------------------------------------------------
# Returns & Risk Metrics
# --------------------------------------------------------------------

def compute_daily_returns(price_df):

    return price_df.pct_change().dropna()


def compute_portfolio_returns(returns_df, weights):

    weights = np.array(weights)

    weights = weights / weights.sum()

    return returns_df.dot(weights)


def annualized_return(daily_returns):

    return daily_returns.mean() * TRADING_DAYS


def annualized_volatility(daily_returns):

    return daily_returns.std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(daily_returns, risk_free_rate=RISK_FREE_RATE):

    ann_return = annualized_return(daily_returns)

    ann_vol = annualized_volatility(daily_returns)

    if ann_vol == 0:

        return 0.0

    return (ann_return - risk_free_rate) / ann_vol


def max_drawdown(cumulative_returns):

    running_max = cumulative_returns.cummax()

    drawdown = (cumulative_returns - running_max) / running_max

    return drawdown.min()


def drawdown_series(cumulative_returns):
    """
    Full underwater curve (not just the min), for plotting drawdown over time.
    """

    running_max = cumulative_returns.cummax()

    return (cumulative_returns - running_max) / running_max


def rolling_volatility(daily_returns, window=30):
    """
    Rolling annualized volatility, useful for spotting periods of turbulence.
    """

    return daily_returns.rolling(window).std() * np.sqrt(TRADING_DAYS)


def individual_asset_stats(prices, returns, weights_dict):
    """
    Per-holding breakdown: total/annual return, annual volatility, Sharpe,
    and each asset's contribution to the overall annualized portfolio return
    (weight * that asset's annualized return).
    """

    rows = []

    for ticker in prices.columns:

        asset_returns = returns[ticker]

        total_return = (prices[ticker].iloc[-1] / prices[ticker].iloc[0] - 1) * 100

        ann_return = annualized_return(asset_returns) * 100

        ann_vol = annualized_volatility(asset_returns) * 100

        sharpe = sharpe_ratio(asset_returns)

        weight = weights_dict.get(ticker, 0)

        rows.append({
            "Ticker": ticker,
            "Weight_%": round(weight * 100, 1),
            "Total_Return_%": round(total_return, 2),
            "Annual_Return_%": round(ann_return, 2),
            "Annual_Volatility_%": round(ann_vol, 2),
            "Sharpe_Ratio": round(sharpe, 3),
            "Contribution_%": round(weight * ann_return, 2),
        })

    df = pd.DataFrame(rows)

    df.sort_values("Weight_%", ascending=False, inplace=True)

    df.reset_index(drop=True, inplace=True)

    return df


def portfolio_summary(tickers, weights=None):
    """
    Full summary for a weighted basket of tickers.
    """

    prices = load_close_prices(tickers)

    if prices.empty:

        print("No data available for the given tickers.")

        return None

    valid_tickers = list(prices.columns)

    if weights is None:

        weights = [1 / len(valid_tickers)] * len(valid_tickers)

    returns = compute_daily_returns(prices)

    port_returns = compute_portfolio_returns(returns, weights)

    cumulative = (1 + port_returns).cumprod()

    summary = {
        "tickers": valid_tickers,
        "weights": dict(zip(valid_tickers, np.round(np.array(weights) / sum(weights), 4))),
        "annual_return": round(annualized_return(port_returns) * 100, 2),
        "annual_volatility": round(annualized_volatility(port_returns) * 100, 2),
        "sharpe_ratio": round(sharpe_ratio(port_returns), 3),
        "max_drawdown": round(max_drawdown(cumulative) * 100, 2),
        "cumulative_return_%": round((cumulative.iloc[-1] - 1) * 100, 2),
    }

    return summary, prices, returns, port_returns, cumulative


# --------------------------------------------------------------------
# Correlation
# --------------------------------------------------------------------

def correlation_matrix(returns_df):

    return returns_df.corr()


def plot_correlation(returns_df, save=True):

    corr = correlation_matrix(returns_df)

    plt.figure(figsize=(8, 6))

    im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

    plt.colorbar(im)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")

    plt.yticks(range(len(corr.columns)), corr.columns)

    for i in range(len(corr.columns)):

        for j in range(len(corr.columns)):

            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)

    plt.title("Portfolio Holdings Correlation")

    plt.tight_layout()

    if save:

        os.makedirs(GRAPHS_FOLDER, exist_ok=True)

        path = os.path.join(GRAPHS_FOLDER, "portfolio_correlation.png")

        plt.savefig(path, dpi=150)

        print(f"Saved -> {path}")

    plt.show()


# --------------------------------------------------------------------
# Monte Carlo Efficient Frontier
# --------------------------------------------------------------------

def simulate_random_portfolios(returns_df, num_portfolios=5000, risk_free_rate=RISK_FREE_RATE):
    """
    Randomly samples portfolio weights to approximate the efficient
    frontier, and identifies the max-Sharpe and min-volatility portfolios.
    """

    tickers = list(returns_df.columns)

    n = len(tickers)

    results = np.zeros((3, num_portfolios))

    all_weights = np.zeros((num_portfolios, n))

    mean_returns = returns_df.mean() * TRADING_DAYS

    cov_matrix = returns_df.cov() * TRADING_DAYS

    for i in range(num_portfolios):

        weights = np.random.random(n)

        weights /= weights.sum()

        all_weights[i, :] = weights

        port_return = np.dot(weights, mean_returns)

        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        results[0, i] = port_return

        results[1, i] = port_vol

        results[2, i] = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0

    columns = ["Return", "Volatility", "Sharpe"]

    results_df = pd.DataFrame(results.T, columns=columns)

    for idx, ticker in enumerate(tickers):

        results_df[ticker] = all_weights[:, idx]

    max_sharpe_port = results_df.loc[results_df["Sharpe"].idxmax()]

    min_vol_port = results_df.loc[results_df["Volatility"].idxmin()]

    return results_df, max_sharpe_port, min_vol_port


def plot_efficient_frontier(results_df, max_sharpe_port, min_vol_port, save=True):

    plt.figure(figsize=(12, 8))

    sc = plt.scatter(
        results_df["Volatility"],
        results_df["Return"],
        c=results_df["Sharpe"],
        cmap="viridis",
        s=10,
        alpha=0.6
    )

    plt.colorbar(sc, label="Sharpe Ratio")

    plt.scatter(
        max_sharpe_port["Volatility"],
        max_sharpe_port["Return"],
        marker="*",
        color="red",
        s=400,
        label="Max Sharpe Ratio"
    )

    plt.scatter(
        min_vol_port["Volatility"],
        min_vol_port["Return"],
        marker="*",
        color="blue",
        s=400,
        label="Min Volatility"
    )

    plt.title("Efficient Frontier (Random Portfolio Simulation)")

    plt.xlabel("Annualized Volatility")

    plt.ylabel("Annualized Return")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    if save:

        os.makedirs(GRAPHS_FOLDER, exist_ok=True)

        path = os.path.join(GRAPHS_FOLDER, "efficient_frontier.png")

        plt.savefig(path, dpi=150)

        print(f"Saved -> {path}")

    plt.show()


# --------------------------------------------------------------------
# Report
# --------------------------------------------------------------------

def save_portfolio_report(summary):

    os.makedirs(REPORTS_FOLDER, exist_ok=True)

    lines = [
        "=" * 50,
        "PORTFOLIO ANALYSIS REPORT",
        "=" * 50,
        f"Holdings           : {', '.join(summary['tickers'])}",
        f"Weights            : {summary['weights']}",
        f"Annual Return      : {summary['annual_return']}%",
        f"Annual Volatility  : {summary['annual_volatility']}%",
        f"Sharpe Ratio       : {summary['sharpe_ratio']}",
        f"Max Drawdown       : {summary['max_drawdown']}%",
        f"Total Return       : {summary['cumulative_return_%']}%",
        "=" * 50
    ]

    text = "\n".join(lines)

    print("\n" + text)

    path = os.path.join(REPORTS_FOLDER, "portfolio_summary.txt")

    with open(path, "w") as f:
        f.write(text)

    print(f"\nSaved -> {path}")


# --------------------------------------------------------------------
# CLI Menu
# --------------------------------------------------------------------

def _select_tickers():

    files = [f[:-4] for f in os.listdir(PROCESSED_FOLDER) if f.endswith(".csv")]

    if not files:

        print("No processed data found. Run Preprocessing first.")

        return []

    print("\nAvailable Companies:", ", ".join(files))

    raw = input("Enter tickers for your portfolio (comma separated, blank = all): ").upper().strip()

    if not raw:

        return files

    tickers = [t.strip() for t in raw.split(",") if t.strip() in files]

    return tickers


def portfolio_menu():

    while True:

        print("\n" + "=" * 55)
        print("        PORTFOLIO ANALYSIS MENU")
        print("=" * 55)

        print("1. Portfolio Summary (Equal Weight)")
        print("2. Portfolio Summary (Custom Weights)")
        print("3. Correlation Matrix")
        print("4. Efficient Frontier (Monte Carlo)")
        print("0. Back")

        choice = input("\nEnter your choice : ")

        if choice == "1":

            tickers = _select_tickers()

            if len(tickers) < 1:
                continue

            result = portfolio_summary(tickers)

            if result:

                summary, prices, returns, port_returns, cumulative = result

                save_portfolio_report(summary)

        elif choice == "2":

            tickers = _select_tickers()

            if len(tickers) < 1:
                continue

            weights = []

            for t in tickers:

                w = float(input(f"Weight for {t} (e.g. 0.5): "))

                weights.append(w)

            result = portfolio_summary(tickers, weights)

            if result:

                summary, prices, returns, port_returns, cumulative = result

                save_portfolio_report(summary)

        elif choice == "3":

            tickers = _select_tickers()

            if len(tickers) < 2:

                print("Correlation requires at least 2 companies.")

                continue

            prices = load_close_prices(tickers)

            returns = compute_daily_returns(prices)

            plot_correlation(returns)

        elif choice == "4":

            tickers = _select_tickers()

            if len(tickers) < 2:

                print("Efficient Frontier requires at least 2 companies.")

                continue

            prices = load_close_prices(tickers)

            returns = compute_daily_returns(prices)

            results_df, max_sharpe_port, min_vol_port = simulate_random_portfolios(returns)

            print("\nMax Sharpe Portfolio:\n", max_sharpe_port)

            print("\nMin Volatility Portfolio:\n", min_vol_port)

            plot_efficient_frontier(results_df, max_sharpe_port, min_vol_port)

        elif choice == "0":

            break

        else:

            print("Invalid Choice")


if __name__ == "__main__":

    portfolio_menu()
