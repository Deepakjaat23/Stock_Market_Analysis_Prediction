"""
preprocessing.py
----------------
Cleans and preprocesses stock market datasets.
"""

import os
import pandas as pd

RAW_FOLDER = "data/raw"
PROCESSED_FOLDER = "data/processed"


def load_stock_data(file_path):
    """
    Load a stock CSV file.
    """
    return pd.read_csv(file_path)


def clean_stock_data(df):
    """
    Clean stock dataset.
    """

    print("\nCleaning dataset...")

    # ----------------------------
    # Convert Date
    # ----------------------------
    df["Date"] = pd.to_datetime(df["Date"])

    # ----------------------------
    # Remove Duplicate Rows
    # ----------------------------
    duplicate_rows = df.duplicated().sum()

    if duplicate_rows > 0:
        print(f"Removed {duplicate_rows} duplicate rows")

    df.drop_duplicates(inplace=True)

    # ----------------------------
    # Missing Values
    # ----------------------------
    print("\nMissing Values")

    print(df.isnull().sum())

    # Remove missing values

    df.dropna(inplace=True)

    # ----------------------------
    # Sort Data
    # ----------------------------
    df.sort_values("Date", inplace=True)

    df.reset_index(drop=True, inplace=True)

    # ----------------------------
    # Date Features
    # ----------------------------

    df["Year"] = df["Date"].dt.year

    df["Month"] = df["Date"].dt.month

    df["Day"] = df["Date"].dt.day

    df["Day_Name"] = df["Date"].dt.day_name()

    df["Weekday"] = df["Date"].dt.dayofweek

    print("\nCleaning Completed")

    return df


def save_processed_data(df, company):
    """
    Save cleaned dataframe.
    """

    os.makedirs(PROCESSED_FOLDER, exist_ok=True)

    file_path = os.path.join(PROCESSED_FOLDER, f"{company}.csv")

    df.to_csv(file_path, index=False)

    print(f"Saved -> {file_path}")


def preprocess_all_data():
    """
    Preprocess every CSV inside data/raw
    """

    processed_data = {}

    files = os.listdir(RAW_FOLDER)

    csv_files = [f for f in files if f.endswith(".csv")]

    if len(csv_files) == 0:

        print("No CSV files found.")

        return processed_data

    print(f"\nFound {len(csv_files)} files.")

    for file in csv_files:

        company = file.replace(".csv", "")

        print("\n-----------------------------------")
        print(company)

        path = os.path.join(RAW_FOLDER, file)

        df = load_stock_data(path)

        df = clean_stock_data(df)

        save_processed_data(df, company)

        processed_data[company] = df

    return processed_data


if __name__ == "__main__":

    data = preprocess_all_data()

    print("\nProcessed Companies")

    print(list(data.keys()))