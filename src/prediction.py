"""
prediction.py
--------------
Stock Market Prediction Module

Trains several regression models on the latest processed dataset,
automatically picks the best-performing one (lowest RMSE), and uses
it to predict tomorrow's close, the next 7 days, and the next 30 days.

Because training always runs against whatever is currently in
data/processed, the "best model" refreshes itself every time you
retrain - if the dataset changes, a different model can win.
"""

import os

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.model_utils import (
    FEATURE_COLUMNS,
    load_dataset,
    prepare_data,
    split_data,
    train_model,
    compare_models,
    plot_model_comparison,
    save_best_model_summary,
    load_model
)


#-----------------------------------------------------Folder Path--------------------------------------------

PROCESSED_FOLDER = "data/processed"


#-----------------------------------------------Select Company---------------------------------------------------

def select_company():

    files = [
        file for file in os.listdir(PROCESSED_FOLDER)
        if file.endswith(".csv")
    ]

    if len(files) == 0:

        print("No processed files found.")

        return None

    print("\nAvailable Companies\n")

    for i, file in enumerate(files, start=1):

        print(f"{i}. {file[:-4]}")

    while True:

        try:

            choice = int(input("\nSelect Company : "))

            if 1 <= choice <= len(files):

                return os.path.join(
                    PROCESSED_FOLDER,
                    files[choice-1]
                )

            print("Invalid Choice")

        except ValueError:

            print("Enter valid number")


#------------------------------------------------Train All Models-------------------------------------------------

def train_all_models(file_path):
    """
    Train 4 models (Linear Regression, Decision Tree, Random Forest,
    XGBoost) on the given processed CSV, compare them, and return the
    name of the best model along with the dataframe used.
    """

    company = os.path.splitext(os.path.basename(file_path))[0]

    df = load_dataset(file_path)

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    models = {

        "Linear Regression": LinearRegression(),

        "Decision Tree": DecisionTreeRegressor(
            random_state=42
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ),

        "XGBoost": XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            random_state=42
        )

    }

    results = []

    for model_name, model in models.items():

        result = train_model(
            model,
            model_name,
            X_train,
            X_test,
            y_train,
            y_test
        )

        results.append(result)

    best_model = compare_models(results)

    print(f"\nBest Model : {best_model}")

    plot_model_comparison(results, company)

    save_best_model_summary(best_model, results, company)

    return best_model, df


#--------------------------------------------------------------------------------------------------------------------

def _get_latest_features(df):
    """
    Helper: returns the feature row for the most recent day.
    """

    return df.iloc[-1:][FEATURE_COLUMNS]


# ==========================================
# TOMORROW PREDICTION
# ==========================================

def predict_tomorrow(best_model_name, df):

    print("\nPredicting Tomorrow's Closing Price...")

    model_file = best_model_name.replace(
        " ",
        "_"
    ).lower() + ".pkl"

    model = load_model(model_file)

    latest = df.iloc[-1:]

    features = latest[FEATURE_COLUMNS]

    prediction = model.predict(features)[0]

    current_price = latest["Close"].values[0]

    change = prediction - current_price

    percent = (change/current_price)*100

    print("\n====================================")

    print(f"Current Close : {current_price:.2f}")

    print(f"Tomorrow Close : {prediction:.2f}")

    print(f"Expected Change : {percent:.2f}%")

    if percent > 1:

        print("Recommendation : BUY")

    elif percent < -1:

        print("Recommendation : SELL")

    else:

        print("Recommendation : HOLD")

    print("====================================")

    return prediction


# ==========================================
# MULTI-DAY PREDICTION (shared logic)
# ==========================================

def _predict_next_n_days(best_model_name, df, days):
    """
    Iteratively predicts `days` days into the future by feeding each
    day's prediction back in as the next day's "Close" feature.
    Note: this is an approximation since only Close is rolled forward
    while the other indicators (MA, RSI, etc.) stay frozen at their
    last known values - it's a simple, explainable forecast, not a
    substitute for a proper time-series model.
    """

    model = load_model(
        best_model_name.replace(
            " ",
            "_"
        ).lower() + ".pkl"
    )

    latest = df.iloc[-1:].copy()

    predictions = []

    print(f"\n{days} Day(s) Prediction\n")

    for day in range(1, days + 1):

        features = latest[FEATURE_COLUMNS]

        pred = model.predict(features)[0]

        predictions.append(pred)

        print(f"Day {day} : {pred:.2f}")

        latest["Close"] = pred

    return predictions


def predict_next_7_days(best_model_name, df):

    return _predict_next_n_days(best_model_name, df, 7)


def predict_next_30_days(best_model_name, df):

    return _predict_next_n_days(best_model_name, df, 30)


# ==========================================
# PREDICTION MENU
# ==========================================

def prediction_menu():

    best_model = None

    dataframe = None

    while True:

        print("\n"+"="*50)

        print("STOCK PREDICTION MENU")

        print("="*50)

        print("1. Train All Models")

        print("2. Tomorrow Prediction")

        print("3. Next 7 Days")

        print("4. Next 30 Days")

        print("0. Back")

        choice = input("\nEnter Choice : ")

        if choice=="1":

            company = select_company()

            if company:

                best_model,dataframe = train_all_models(company)

        elif choice=="2":

            if best_model is None:

                print("Train Models First.")

            else:

                predict_tomorrow(best_model,dataframe)

        elif choice=="3":

            if best_model is None:

                print("Train Models First.")

            else:

                predict_next_7_days(best_model,dataframe)

        elif choice=="4":

            if best_model is None:

                print("Train Models First.")

            else:

                predict_next_30_days(best_model,dataframe)

        elif choice=="0":

            break

        else:

            print("Invalid Choice")


if __name__=="__main__":

    prediction_menu()
