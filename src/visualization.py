"""
visualization.py
----------------
Visualizes stock market data.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import mplfinance as mpf
from scipy.stats import norm

PROCESSED_FOLDER = "data/processed"

plt.style.use("ggplot")


#-------------------------------------------LOAD ALL DATA-------------------------------------------------

def load_all_data():
    """
    Load all processed CSV files.
    """

    datasets = {}

    files = [f for f in os.listdir(PROCESSED_FOLDER) if f.endswith(".csv")]

    for file in files:

        company = file.replace(".csv", "")

        path = os.path.join(PROCESSED_FOLDER, file)

        datasets[company] = pd.read_csv(path, parse_dates=["Date"])

    return datasets


#-------------------------------------------------PRICE TREND-----------------------------------------------

def plot_price_trend(datasets):

    plt.figure(figsize=(14,7))

    for company, df in datasets.items():

        plt.plot(
            df["Date"],
            df["Close"],
            linewidth=2,
            label=company
        )

    plt.title("Stock Price Trend", fontsize=18)

    plt.xlabel("Date")

    plt.ylabel("Closing Price")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.show()


#------------------------------------------------VOLUME TREND-------------------------------------------------

def plot_volume(datasets):

    plt.figure(figsize=(14,7))

    for company, df in datasets.items():

        plt.plot(
            df["Date"],
            df["Volume"],
            label=company
        )

    plt.title("Trading Volume")

    plt.xlabel("Date")

    plt.ylabel("Volume")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.show()


#------------------------------------------------MOVING AVRRAGE----------------------------------------------

def plot_moving_average(datasets):

    for company, df in datasets.items():

        plt.figure(figsize=(14,7))

        plt.plot(df["Date"], df["Close"], label="Close Price")

        plt.plot(df["Date"], df["MA20"], label="MA20")

        plt.plot(df["Date"], df["MA50"], label="MA50")

        plt.plot(df["Date"], df["MA100"], label="MA100")

        plt.title(f"{company} Moving Average")

        plt.xlabel("Date")

        plt.ylabel("Price")

        plt.legend()

        plt.grid(True)

        plt.tight_layout()

        plt.show()


#--------------------------------------------Bollinger Bands------------------------------------------------

def plot_bollinger_bands(datasets):

    for company, df in datasets.items():

        plt.figure(figsize=(14,7))

        plt.plot(df["Date"], df["Close"], label="Close")

        plt.plot(df["Date"], df["BB_Upper"], linestyle="--", label="Upper Band")

        plt.plot(df["Date"], df["BB_Lower"], linestyle="--", label="Lower Band")

        plt.fill_between(
            df["Date"],
            df["BB_Upper"],
            df["BB_Lower"],
            alpha=0.2
        )

        plt.title(f"{company} Bollinger Bands")

        plt.legend()

        plt.grid(True)

        plt.tight_layout()

        plt.show()


#--------------------------------------------------------RSI-------------------------------------------------

def plot_rsi(datasets):

    for company, df in datasets.items():

        plt.figure(figsize=(14,6))

        plt.plot(df["Date"], df["RSI"], color="purple", label="RSI")

        plt.axhline(70, color="red", linestyle="--", label="Overbought")

        plt.axhline(30, color="green", linestyle="--", label="Oversold")

        plt.title(f"{company} RSI (Relative Strength Index)")

        plt.xlabel("Date")

        plt.ylabel("RSI")

        plt.legend()

        plt.grid(True)

        plt.tight_layout()

        plt.show()


#--------------------------------------------------------MACD--------------------------------------------------

def plot_macd(datasets):

    for company, df in datasets.items():

        plt.figure(figsize=(14,6))

        plt.plot(df["Date"], df["MACD"], label="MACD")

        plt.plot(df["Date"], df["Signal_Line"], label="Signal Line")

        plt.title(f"{company} MACD")

        plt.xlabel("Date")

        plt.ylabel("MACD")

        plt.legend()

        plt.grid(True)

        plt.tight_layout()

        plt.show()


#-------------------------------------------------DAILY RETURN------------------------------------------------

def plot_daily_return(datasets):

    plt.figure(figsize=(14,6))

    for company, df in datasets.items():

        plt.plot(
            df["Date"],
            df["Daily_Return"],
            label=company
        )

    plt.title("Daily Return Comparison")

    plt.xlabel("Date")

    plt.ylabel("Daily Return")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.show()


#-----------------------------------------------ROLLING VOLATILITY-------------------------------------------

def plot_volatility(datasets):

    plt.figure(figsize=(14,6))

    for company, df in datasets.items():

        plt.plot(
            df["Date"],
            df["Volatility"],
            label=company
        )

    plt.title("Rolling Volatility")

    plt.xlabel("Date")

    plt.ylabel("Volatility")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.show()


#----------------------------------------------CUMULATIVE RETURN----------------------------------------------

def plot_cumulative_return(datasets):

    plt.figure(figsize=(14,6))

    for company, df in datasets.items():

        plt.plot(
            df["Date"],
            df["Cumulative_Return"],
            linewidth=2,
            label=company
        )

    plt.title("Cumulative Return Comparison")

    plt.xlabel("Date")

    plt.ylabel("Cumulative Return")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.show()


#-----------------------------------------CORRELATION HEATMAP------------------------------------------------

def plot_correlation_heatmap(datasets):

    for company, df in datasets.items():

        plt.figure(figsize=(10,8))

        corr = df[
            [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume"
            ]
        ].corr()

        sns.heatmap(
            corr,
            annot=True,
            cmap="coolwarm",
            linewidths=0.5
        )

        plt.title(f"{company} Correlation Heatmap")

        plt.tight_layout()

        plt.show()


#-----------------------------------------------RISK VS RETURN-----------------------------------------------

def plot_risk_return(datasets):

    plt.figure(figsize=(12,8))

    for company, df in datasets.items():

        risk = df["Daily_Return"].std()

        returns = df["Daily_Return"].mean()

        plt.scatter(
            risk,
            returns,
            s=200
        )

        plt.text(
            risk,
            returns,
            company,
            fontsize=10
        )

    plt.title("Risk vs Return")

    plt.xlabel("Risk (Std Dev)")

    plt.ylabel("Average Return")

    plt.grid(True)

    plt.tight_layout()

    plt.show()


#-------------------------------------------DAILY HIGH LOW SPREAD----------------------------------------------

def plot_daily_spread(datasets):

    plt.figure(figsize=(14,7))

    for company, df in datasets.items():

        plt.plot(
            df["Date"],
            df["High_Low_Spread"],
            label=company
        )

    plt.title("Daily High-Low Spread")

    plt.xlabel("Date")

    plt.ylabel("Spread")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.show()


#----------------------------------------------VOLUME VS DAILY RETURN-----------------------------------------

def plot_volume_vs_return(datasets):

    for company, df in datasets.items():

        plt.figure(figsize=(10,6))

        plt.scatter(
            df["Daily_Return"],
            df["Volume"],
            alpha=0.5
        )

        plt.title(f"{company} Volume vs Daily Return")

        plt.xlabel("Daily Return")

        plt.ylabel("Volume")

        plt.grid(True)

        plt.tight_layout()

        plt.show()


#--------------------------------------------MONTHLY RETURN HEATMAP-----------------------------------------

def plot_monthly_return_heatmap(datasets):

    for company, df in datasets.items():

        temp = df.copy()

        temp["Year"] = pd.to_datetime(temp["Date"]).dt.year

        temp["Month"] = pd.to_datetime(temp["Date"]).dt.month

        pivot = temp.pivot_table(
            values="Daily_Return",
            index="Year",
            columns="Month",
            aggfunc="mean"
        )

        plt.figure(figsize=(12,6))

        sns.heatmap(
            pivot,
            cmap="RdYlGn",
            annot=True,
            fmt=".2%"
        )

        plt.title(f"{company} Monthly Return Heatmap")

        plt.tight_layout()

        plt.show()


#--------------------------------------------Candlestick Chart-----------------------------------------------

def plot_candlestick(datasets):

    for company, df in datasets.items():

        temp = df.copy()

        temp["Date"] = pd.to_datetime(temp["Date"])

        temp.set_index("Date", inplace=True)

        ohlc = temp[
            ["Open", "High", "Low", "Close", "Volume"]
        ]

        mpf.plot(
            ohlc.tail(90),
            type="candle",
            style="charles",
            volume=True,
            mav=(20,50),
            title=f"{company} Candlestick (Last 90 Days)"
        )


#----------------------------------------------RELATIVE STRENGTH--------------------------------------------

def plot_relative_strength(datasets):

    companies = list(datasets.keys())

    if len(companies) < 2:

        print("Relative Strength requires at least 2 companies.")

        return

    benchmark = companies[0]

    benchmark_df = datasets[benchmark].set_index("Date")

    benchmark_close = benchmark_df["Close"]

    for company in companies[1:]:

        company_close = datasets[company].set_index("Date")["Close"]

        relative = (
            company_close /
            benchmark_close
        ).dropna()

        plt.figure(figsize=(14,6))

        plt.plot(relative.index, relative.values)

        plt.title(f"{company} Relative to {benchmark}")

        plt.grid(True)

        plt.tight_layout()

        plt.show()


#------------------------------------------------ROLLING VOLATILITY--------------------------------------------

def plot_rolling_volatility(datasets):

    for company, df in datasets.items():

        rolling = (
            df["Daily_Return"]
            .rolling(30)
            .std()
            * np.sqrt(252)
        )

        plt.figure(figsize=(14,6))

        plt.plot(df["Date"], rolling)

        plt.title(f"{company} Rolling Volatility")

        plt.grid(True)

        plt.tight_layout()

        plt.show()


#-------------------------------------------Monte Carlo Simulation-----------------------------------------------

def monte_carlo_simulation(datasets, days=30):

    for company, df in datasets.items():

        returns = df["Daily_Return"].dropna()

        mu = returns.mean()

        sigma = returns.std()

        last_price = df["Close"].iloc[-1]

        plt.figure(figsize=(12,6))

        for _ in range(100):

            prices = [last_price]

            for _ in range(days):

                shock = np.random.normal(mu, sigma)

                prices.append(
                    prices[-1] * (1 + shock)
                )

            plt.plot(prices, alpha=0.2)

        plt.title(f"{company} Monte Carlo")

        plt.grid(True)

        plt.tight_layout()

        plt.show()


#--------------------------------------------Closing Price Distribution--------------------------------------------

def plot_distribution(datasets):

    for company, df in datasets.items():

        plt.figure(figsize=(10,6))

        sns.kdeplot(
            df["Close"],
            fill=True
        )

        plt.title(f"{company} Closing Price Distribution")

        plt.grid(True)

        plt.tight_layout()

        plt.show()


#------------------------------------------CLOSING PRICE HISTOGRAM-------------------------------------------

def plot_histogram(datasets):

    for company, df in datasets.items():

        plt.figure(figsize=(10,6))

        plt.hist(
            df["Close"],
            bins=30
        )

        plt.title(f"{company} Closing Price Histogram")

        plt.grid(True)

        plt.tight_layout()

        plt.show()







# ==========================================
# MASTER FUNCTION
# ==========================================

def show_all_visualizations():

    datasets = load_all_data()


    # Part 1

    plot_price_trend(datasets)

    plot_volume(datasets)

    plot_moving_average(datasets)

    plot_bollinger_bands(datasets)


    # Part 2

    plot_rsi(datasets)

    plot_macd(datasets)

    plot_daily_return(datasets)

    plot_volatility(datasets)

    plot_cumulative_return(datasets)


    # Part 3

    plot_correlation_heatmap(datasets)

    plot_risk_return(datasets)

    plot_daily_spread(datasets)

    plot_volume_vs_return(datasets)

    plot_monthly_return_heatmap(datasets)


    # Part 4

    plot_candlestick(datasets)

    plot_relative_strength(datasets)

    plot_rolling_volatility(datasets)

    monte_carlo_simulation(datasets)

    plot_distribution(datasets)

    plot_histogram(datasets)


#-----------------------------------------------Menu Folder--------------------------------------------------

def visualization_menu():

    datasets = load_all_data()

    while True:

        print("\n" + "="*55)
        print("        STOCK VISUALIZATION MENU")
        print("="*55)

        print("1. Price Trend")
        print("2. Trading Volume")
        print("3. Moving Average")
        print("4. Bollinger Bands")
        print("5. RSI")
        print("6. MACD")
        print("7. Daily Return")
        print("8. Rolling Volatility")
        print("9. Cumulative Return")
        print("10. Correlation Heatmap")
        print("11. Risk vs Return")
        print("12. Daily High-Low Spread")
        print("13. Volume vs Daily Return")
        print("14. Monthly Return Heatmap")
        print("15. Candlestick Chart")
        print("16. Relative Strength")
        print("17. Monte Carlo Simulation")
        print("18. Closing Price Distribution")
        print("19. Closing Price Histogram")
        print("20. Show All Graphs")
        print("0. Exit")

        choice = input("\nEnter your choice : ")

        if choice == "1":
            plot_price_trend(datasets)

        elif choice == "2":
            plot_volume(datasets)

        elif choice == "3":
            plot_moving_average(datasets)

        elif choice == "4":
            plot_bollinger_bands(datasets)

        elif choice == "5":
            plot_rsi(datasets)

        elif choice == "6":
            plot_macd(datasets)

        elif choice == "7":
            plot_daily_return(datasets)

        elif choice == "8":
            plot_volatility(datasets)

        elif choice == "9":
            plot_cumulative_return(datasets)

        elif choice == "10":
            plot_correlation_heatmap(datasets)

        elif choice == "11":
            plot_risk_return(datasets)

        elif choice == "12":
            plot_daily_spread(datasets)

        elif choice == "13":
            plot_volume_vs_return(datasets)

        elif choice == "14":
            plot_monthly_return_heatmap(datasets)

        elif choice == "15":
            plot_candlestick(datasets)

        elif choice == "16":
            plot_relative_strength(datasets)

        elif choice == "17":
            monte_carlo_simulation(datasets)

        elif choice == "18":
            plot_distribution(datasets)

        elif choice == "19":
            plot_histogram(datasets)

        elif choice == "20":

            show_all_visualizations()

        elif choice == "0":

            print("Returning...")

            break

        else:

            print("Invalid Choice")


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    visualization_menu()