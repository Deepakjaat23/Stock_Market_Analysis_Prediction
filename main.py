"""
main.py
--------
Main Controller for Stock Market Analysis & Prediction System
"""

from src.download_data import (
    get_stock_symbols,
    download_multiple_stocks
)


from src.preprocessing import preprocess_all_data

from src.feature_engineering import process_all

from src.visualization import visualization_menu

from src.prediction import prediction_menu

from src.deep_learning import deep_learning_menu

from src.portfolio import portfolio_menu

from src.sentiment import sentiment_menu

from src.comparison import comparison_menu

from src.recommendation import recommendation_menu


def display_menu():

    print("\n" + "=" * 60)
    print("      STOCK MARKET ANALYSIS & PREDICTION SYSTEM")
    print("=" * 60)

    print("1. Download Stock Data")

    print("2. Preprocess Data")

    print("3. Feature Engineering")

    print("4. Visualization")

    print("5. Prediction (ML Regression Models)")

    print("6. Deep Learning Forecast (LSTM / Prophet)")

    print("7. Portfolio Analysis")

    print("8. News Sentiment Analysis")

    print("9. Multi-Stock Comparison")

    print("10. Buy/Sell Recommendation")

    print("11. About Project")

    print("0. Exit")

    print("\nTip: run 'streamlit run streamlit_app.py' for the full")
    print("     interactive web dashboard with live data and charts.")


def about():

    print("\nPROJECT DETAILS")

    print("-" * 40)

    print("Project : Stock Market Analysis & Prediction")

    print("Language : Python")

    print("Data Source : Yahoo Finance")

    print("Developer : Deepak")

    print("Features")

    print("✔ Download Stock Data")

    print("✔ Data Cleaning")

    print("✔ Feature Engineering")

    print("✔ Technical Analysis")

    print("✔ Machine Learning Prediction")

    print("✔ Deep Learning Forecasting (LSTM & Prophet)")

    print("✔ Portfolio Analysis (Risk, Sharpe, Efficient Frontier)")

    print("✔ News Sentiment Analysis")

    print("✔ Multi-Stock Comparison")

    print("✔ Buy/Sell Recommendation Engine")

    print("✔ Streamlit Real-Time Web Dashboard")


def main():

    while True:

        display_menu()

        choice = input("\nEnter your choice : ")

        if choice == "1":

            tickers = get_stock_symbols()

            download_multiple_stocks(tickers)

            print("\nData Download Completed.")

        elif choice == "2":

            preprocess_all_data()

            print("\nPreprocessing Completed.")

        elif choice == "3":

            process_all()

            print("\nFeature Engineering Completed.")

        elif choice == "4":

            visualization_menu()

        elif choice == "5":

            prediction_menu()

        elif choice == "6":

            deep_learning_menu()

        elif choice == "7":

            portfolio_menu()

        elif choice == "8":

            sentiment_menu()

        elif choice == "9":

            comparison_menu()

        elif choice == "10":

            recommendation_menu()

        elif choice == "11":

            about()

        elif choice == "0":

            print("\nThank You")

            print("Exiting Project...")

            break

        else:

            print("\nInvalid Choice")


if __name__ == "__main__":

    main()

