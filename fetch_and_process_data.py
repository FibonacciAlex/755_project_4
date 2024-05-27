# Module Imports:
import pandas as pd
import os
import json
import math
from datetime import datetime
import pytz
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import pprint

from FetchData import FetchData


# Functions:


def engineer_features(df, selected_item_trends):
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize(pytz.utc)
    df = df.groupby(df.index).mean()

    selected_item_trends.index = selected_item_trends.index.tz_localize(pytz.utc)
    df = pd.merge(df, selected_item_trends, left_index=True, right_index=True, how='inner')


    today = datetime.now(pytz.utc)
    df["days_old"] = (today - df.index).days
    df = df[df['days_old'] != 0]

    df["daily_views"] = df["views"] // df["days_old"]
    df["daily_likes"] = df["likes"] // df["days_old"]
    df["daily_comments"] = df["comments"] // df["days_old"]

    df["daily_likes_to_views_ratio"] = df["daily_likes"] // df["daily_views"]
    df["daily_comments_to_views_ratio"] = df["daily_comments"] // df["daily_views"]
    df["trend_to_daily_views_ratio"] = df["trend"] // df["daily_views"]
    df["trend_to_daily_likes_ratio"] = df["trend"] // df["daily_likes"]


    df['diff_daily_views'] = df['daily_views'].diff()
    df['diff_daily_likes'] = df['daily_likes'].diff()
    df['diff_daily_comments'] = df['daily_comments'].diff()

    # Replace all null and infinity values:
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)


    # Normalise data:
    df_to_normalize = df.drop(columns=['trend'])
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df_to_normalize), columns=df_to_normalize.columns, index=df.index)
    df_normalized['trend'] = df["trend"]
    return df_normalized

def custom_distance(x, y):
    # Calculate the Euclidean distance:
    euclidean_distance = np.sqrt(np.sum((x - y)**2))
    # Clip the distance based on the range of the target variable:
    clipped_distance = np.clip(euclidean_distance, 0, 100)
    return clipped_distance

def replace_nan_with_null(obj):
    if isinstance(obj, dict):
        return {k: replace_nan_with_null(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_nan_with_null(elem) for elem in obj]
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    else:
        return obj


def convert_output_to_dict(df):
    df.reset_index(inplace=True)
    df.rename(columns={"index": "date"}, inplace=True)
    df["date"] = df["date"].astype(str)
    dict_data = df.to_dict(orient="records")
    return dict_data

def run_model(model_type, df, features, target, plotModel=True):
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, shuffle=False, random_state=None)

    # Initialize and train the model using a train test split:
    model = 0
    if model_type == "knn":
        model = KNeighborsRegressor(n_neighbors=2, metric=custom_distance)
    elif model_type == "linearregression":
        model = LinearRegression()
    elif "decisiontree":
        model = DecisionTreeRegressor(random_state=1)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred = np.round(y_pred).astype(int)
    scaler = MinMaxScaler(feature_range=(0, 100))
    
    #print(y_pred)
    y_pred = pd.Series(y_pred)
    y_pred.index = y_test.index
    #print(y_test)
    #print(y_pred)
    mse = mean_squared_error(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print("MSE:", mse)
    print("Accuracy:", accuracy)

    
    accuracy_df = pd.DataFrame(y_test)
    accuracy_df.columns = ["y_test"]
    accuracy_df["y_pred"] = y_pred
    accuracy_df['y_pred'] = accuracy_df['y_pred'].clip(0, 100)

    if plotModel:
        plt.plot(accuracy_df["y_test"], label="Actual", linestyle="-", color="blue",)
        plt.plot(accuracy_df["y_pred"], label="Predicted", linestyle="--", color="orange",)
        plt.xlabel("Date")
        plt.ylabel("Trend")
        plt.title("Actual vs. Predicted Trend")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # Predict the next 7 days
    # Use the last 7 days of the dataset as features for prediction
    last_7_days = X.tail(7)
    pred_next_7_days = model.predict(last_7_days)
    pred_next_7_days = np.round(pred_next_7_days).astype(int)

    # Create a DataFrame for the next 7 days predictions
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=7, freq="D")
    pred_df = pd.DataFrame(pred_next_7_days, index=future_dates, columns=[target])
    pred_df = pd.concat([df, pred_df])
    pred_df.rename(columns={target: "trend_pred"}, inplace=True)

    # Pad the original data with 7 more days of empty data
    padding_dates = future_dates
    padding_df = pd.DataFrame(index=padding_dates, columns=df.columns)
    df_padded = pd.concat([df, padding_df])

    merged_df = df_padded[["trend"]].merge(pred_df[["trend_pred"]], left_index=True, right_index=True)
    merged_df['trend_pred'] = merged_df['trend_pred'].clip(0, 100)

    if plotModel:
        plt.plot(merged_df["trend"], linestyle="-", color="blue", label="Real")
        plt.plot(merged_df["trend_pred"], linestyle="--", color="orange", label="Pred")
        plt.xlabel("Date")
        plt.ylabel("Trend")
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.show()

    return merged_df



# Main Process:



data_fetcher = FetchData(False, False)

news_items = (data_fetcher.read_news_items())["news_items"].tolist()
#data_fetcher.fetch_yt_videos_data(news_items)

trends_df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "./trends_data.csv"))
#trends_df = data_fetcher.fetch_trends_data(news_items)
trends_df["date"] = pd.to_datetime(trends_df["date"]).dt.date


# Create and Run Models:
model_output_dict = {}
for i, item in enumerate(news_items):
    model_output_dict[i] = {
        "knn": "",
        "linearregression": "",
        "decisiontree": ""
    }

    # Prepare Data:
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"./yt_data_{i}.csv"))
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.drop(columns=["id"])
    df.set_index("date", inplace=True)
    df = df.sort_index()

    selected_item_trends = trends_df[['date', trends_df.columns[i+1]]]
    selected_item_trends["date"] = pd.to_datetime(selected_item_trends["date"])
    selected_item_trends.set_index("date", inplace=True)
    selected_item_trends.rename(columns={selected_item_trends.columns[0]: "trend"}, inplace=True)


    df = engineer_features(df, selected_item_trends)

    print(df)

    features = [
        "views",
        #"likes",
        #"comments",
        "daily_views",
        #"daily_likes",
        #"daily_comments",
        "diff_daily_views",
        #"diff_daily_likes",
        "daily_likes_to_views_ratio",
        "daily_comments_to_views_ratio",
        "trend_to_daily_views_ratio",
        "trend_to_daily_likes_ratio"
        ]
    target = "trend"

    # kNN:
    knn_result = run_model("knn", df, features, target, True)
    # Save the kNN predictions to the dict:
    dict_data = convert_output_to_dict(knn_result)
    model_output_dict[i]["knn"] = dict_data
    
    # Linear Regression:
    linear_regression_result = run_model("linearregression", df, features, target, False)
    # Save the kNN predictions to the dict:
    dict_data = convert_output_to_dict(linear_regression_result)
    model_output_dict[i]["linearregression"] = dict_data

    # Decision Tree Regresssion:
    decision_regression_result = run_model("decisiontree", df, features, target, True)
    # Save the kNN predictions to the dict:
    dict_data = convert_output_to_dict(decision_regression_result)
    model_output_dict[i]["decisiontree"] = dict_data



# Save all model outputs to json:
dict_with_nulls = replace_nan_with_null(model_output_dict)

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "./model_data.json"), "w") as json_file:
    json.dump(dict_with_nulls, json_file, indent=4)