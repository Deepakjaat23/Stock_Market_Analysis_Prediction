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


def display_menu():

    print("\n" + "=" * 60)
    print("      STOCK MARKET ANALYSIS & PREDICTION SYSTEM")
    print("=" * 60)

    print("1. Download Stock Data")

    print("2. Preprocess Data")

    print("3. Feature Engineering")

    print("4. Visualization")

    print("5. Prediction")

    print("6. About Project")

    print("0. Exit")


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

            about()

        elif choice == "0":

            print("\nThank You")

            print("Exiting Project...")

            break

        else:

            print("\nInvalid Choice")


if __name__ == "__main__":

    main()

