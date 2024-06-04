import pandas as pd
import requests
from pytrends.request import TrendReq
from datetime import datetime, timedelta, timezone
import time
import random
import holidays
import math
import numpy as np
import pytz
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

class FetchData:

    # Path & URL Strings:
    def __init__(self, loader: any):
        self._yt_api = "https://yt.lemnoslife.com/noKey/"
        self.loader = loader
    
    @property
    def yt_api(self):
        return self._yt_api
    @yt_api.setter
    def yt_api(self, new_value):
        self._yt_api = new_value

    def fetch_trends_data(self, news_item: str):
        #pd.set_option("future.no_silent_downcasting", True) # Hides downcasting warning

        trends_df = pd.DataFrame()

        pytrends = TrendReq(retries=30, backoff_factor=0.1, hl='en-US')

        pytrends.build_payload([news_item], cat=0, timeframe="today 1-m", geo='', gprop='')
        df = pytrends.interest_over_time()
        df = df.rename_axis("date").reset_index()
        df["date"] = pd.to_datetime(df["date"])
        df = df.drop(columns="isPartial")
        if len(trends_df) == 0:
            trends_df = df
        else:
            trends_df = pd.merge(trends_df, df, on="date", how="inner")

        # Convert all columns to int except "date"
        num_columns = df.columns.difference(["date"])
        trends_df[num_columns] = df[num_columns].astype(int)

        return trends_df


    def fetch_yt_videos_data(self, news_item: str, days_to_fetch = 30, videos_to_fetch = 1):

        progress_increment = max(1, math.floor(((days_to_fetch * videos_to_fetch) + 30) / 100))
        progress_percent = 0

        def create_date_ranges(num_days):
            date_ranges = []
            today = datetime.now()
            
            for i in range(num_days):
                start_date = today - timedelta(days=i)
                start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
                end_date = start_date.replace(hour=23, minute=59, second=59, microsecond=0)
                
                start_date_str = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
                end_date_str = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
                
                date_ranges.append({
                    "start_date_str": start_date_str,
                    "end_date_str": end_date_str
                })
            return date_ranges
        
        api = self.yt_api

        list_of_dfs = []

        # Fetch top X video for each day:
        endpoint = "search"
        max_results = videos_to_fetch

        # Get the timestamp for 1 month ago, formatted correctly:
        date_ranges = create_date_ranges(days_to_fetch)
        results = {}


        print(f"Fetching video IDS for '{news_item}'.")
        videos = []
        for date in date_ranges:
            params = {
                "part": "snippet",
                "maxResults": max_results,
                "q": news_item,
                "order": "viewCount",
                "publishedAfter": date["start_date_str"],
                "publishedBefore": date["end_date_str"],
            }
            response = requests.get(api + endpoint, params=params)
            data = response.json()
            for video in data["items"]:
                videos.append({
                    "date": video["snippet"]["publishedAt"],
                    "id": video["id"]["videoId"]
                })

            progress_percent = progress_percent + progress_increment
            if progress_percent >= 97: progress_percent = 97
            self.loader.progress(progress_percent, text=f"Fetching feature data for '{news_item}'.")

        results[news_item] = videos
        print(f"Fetching video IDS for '{news_item}' is complete!.")
        time.sleep(random.uniform(0.01, 0.1))
        
        # Get stats for each items" videos:
        final_results = {}
        for i, item in enumerate(results):
            print(f"Fetching video data for '{item}'.")
            endpoint = "videos"
            video_stats = []
            for j, video in enumerate(results[item]):
                params = {
                    "part": "statistics",
                    "id": video["id"]
                }
                response = requests.get(self.yt_api + endpoint, params=params)
                data = response.json()
                data["metadata"] = {"id": video["id"], "date": video["date"]}
                video_stats.append(data)
                print(f"Video data {j} for item '{item}' complete!.")

                progress_percent = progress_percent + progress_increment
                if progress_percent >= 97: progress_percent = 97
                progress_text = f"Fetching feature data for '{news_item}'."
                if progress_percent > 50:
                    f"Over halfway! Fetching feature data for '{news_item}'."
                if progress_percent > 80:
                    f"Almost there! Fetching feature data for '{news_item}'."
                self.loader.progress(progress_percent+progress_increment, text=progress_text)

            final_results[item] = video_stats


        # Create the dataframes and add them to the list:

        for entry in final_results:
            data = []
            for i, val in enumerate(final_results[entry]):
                views = 0
                likes = 0
                comments = 0
                try:
                    views = int(final_results[entry][i]["items"][0]["statistics"]["viewCount"])
                except:
                    views = 0
                try:
                    likes = int(final_results[entry][i]["items"][0]["statistics"]["likeCount"])
                except:
                    likes = 0
                try:
                    comments = int(final_results[entry][i]["items"][0]["statistics"]["commentCount"])
                except:
                    comments = 0

                data.append({
                    "date": final_results[entry][i]["metadata"]["date"],
                    "id": final_results[entry][i]["metadata"]["id"],
                    "views": views,
                    "likes": likes,
                    "comments": comments
                })
            df = pd.DataFrame(data, index=None)

            list_of_dfs.append(df)

        return list_of_dfs

    def merge_trends_and_yt_data(self, trends_df, yt_data_list):
        final_df_list = []

        for i, df in enumerate(yt_data_list):
            merged_df = df
            merged_df["trend"] = trends_df.iloc[:, i+1]
            final_df_list.append(merged_df)

        return final_df_list

    def fetch_and_return_final_df_list(self, item_string: str):
        trends = self.fetch_trends_data(item_string)
        yt_data_list = self.fetch_yt_videos_data(item_string)
        #final_df_list = self.merge_trends_and_yt_data(trends, yt_data_list)
        
        self.loader.empty()

        return trends, yt_data_list[0]
    

class CreateFeatures:

    def create_calendar_feats(self, chosen_year):
        start_date = f"{str(chosen_year-1)}-01-01"
        end_date = f"{str(chosen_year+1)}-12-31"
        date_range = pd.date_range(start=start_date, end=end_date)
        calendar_df = pd.DataFrame(date_range, columns=["date"])
        calendar_df["day_number"] = calendar_df["date"].dt.dayofweek
        # Add the binary holidays column:
        us_holidays = holidays.US(years=[chosen_year-1, chosen_year+1])
        calendar_df["is_holiday"] = calendar_df["date"].apply(lambda x: 1 if x in us_holidays else 0)
        calendar_df.set_index('date', inplace=True)
        calendar_df.index = pd.to_datetime(calendar_df.index)
        calendar_df.index = calendar_df.index.tz_localize(None)
        calendar_df.index = calendar_df.index.normalize()

        return calendar_df
    
    def create_features(self, trend_df, yt_df):
        
        # Prep Trend data:
        trend_df.set_index("date", inplace=True)
        trend_df.index = pd.to_datetime(trend_df.index)
        trend_df.index = trend_df.index.normalize()
        trend_df.rename(columns={trend_df.columns[0]: "trend"}, inplace=True)

        # Prep YT data:
        if 'id' in yt_df.columns:
            yt_df = yt_df.drop(columns=["id"])
        yt_df.set_index("date", inplace=True)
        yt_df.index = pd.to_datetime(yt_df.index)
        yt_df = yt_df.sort_index()
        yt_df.index = yt_df.index.tz_localize(None)
        yt_df.index = yt_df.index.normalize()
        yt_df = yt_df.groupby(yt_df.index).mean()

        # Create Features:
        df = yt_df.join(trend_df, how='inner')

        # Extrapolate feature data for the last day:
        total_seconds_in_day = 24 * 60 * 60
        current_utc_time = datetime.now(timezone.utc)
        end_of_day_utc = datetime(current_utc_time.year, current_utc_time.month, current_utc_time.day, 23, 59, 59, tzinfo=timezone.utc)
        seconds_remaining = (end_of_day_utc - current_utc_time).total_seconds()
        percentage_remaining = seconds_remaining / total_seconds_in_day

        adjusted_values = df.loc[df.index.max()] * percentage_remaining
        df.loc[df.index.max()] = adjusted_values



        last_day = df.index.max()
        df["days_old"] = (last_day - df.index).days
        df.loc[df['days_old'] == 0, 'days_old'] = 1

        df["daily_views"] = df["views"] // df["days_old"]
        df["daily_likes"] = df["likes"] // df["days_old"]
        df["daily_comments"] = df["comments"] // df["days_old"]

        df["daily_likes_to_views_ratio"] = df["daily_likes"] // df["daily_views"]
        df["daily_comments_to_views_ratio"] = df["daily_comments"] // df["daily_views"]
        df["trend_to_daily_views_ratio"] = df["trend"] // df["daily_views"]
        df["trend_to_daily_likes_ratio"] = df["trend"] // df["daily_likes"]

        df["diff_daily_views"] = df["daily_views"].diff()
        df["diff_daily_likes"] = df["daily_likes"].diff()
        df["diff_daily_comments"] = df["daily_comments"].diff()

        # Replace all null and infinity values:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

        # Normalise data:
        df_to_normalize = df.drop(columns=["trend"])
        scaler = MinMaxScaler()
        df_normalised = pd.DataFrame(scaler.fit_transform(df_to_normalize), columns=df_to_normalize.columns, index=df.index)
        df_normalised["trend"] = df["trend"]

        # Merge the calendar data:
        current_year = datetime.now().year
        calendar_df = self.create_calendar_feats(current_year)
        df = pd.merge(df, calendar_df, left_index=True, right_index=True, how="inner")
        df_normalised = pd.merge(df_normalised, calendar_df, left_index=True, right_index=True, how="inner")

        return df, df_normalised



class RunAnalysis:
    
    def get_corr_matrix(self, df):
        correlation_matrix = df.corr()

        # Extract correlations with the 'trend' column
        correlation_with_trend = correlation_matrix['trend'].drop(labels=['trend'])

        # Convert the correlation series to a DataFrame for easier plotting
        correlation_df = correlation_with_trend.reset_index()
        correlation_df.columns = ['Feature', 'Correlation']
        return correlation_df



class RunModels:

    def run_model(self, model_type, df, features, target):

        def custom_distance(x, y):
            # Calculate the Euclidean distance:
            euclidean_distance = np.sqrt(np.sum((x - y)**2))
            # Clip the distance based on the range of the target variable:
            clipped_distance = np.clip(euclidean_distance, 0, 100)
            return clipped_distance
        

        # Initialize and train the model using a train test split:
        model = None
        last_row_date = df.index.max()

        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False, random_state=None)

        # Split off last (extrapolated) day of data for later prediction!
        last_row_test_X = X_test.loc[[X_test.index.max()]]
        last_row_test_y = y_test.loc[[y_test.index.max()]]
        X_test = X_test.drop(X_test.index.max())
        y_test = y_test.drop(y_test.index.max())
        df = df.drop(df.index.max())
        y_pred = 0;

        if model_type == "knn":
            model = KNeighborsRegressor(n_neighbors=5, metric=custom_distance)
        elif model_type == "linearregression":
            model = LinearRegression()
        elif model_type == "randomforest":
            model = RandomForestRegressor(n_estimators=10, random_state=0)

        if model == None and model_type == "lightgbm":
            train_data = lgb.Dataset(X_train, label=y_train)
            params = {
                'objective': 'regression',
                'metric': 'mse',
                'num_leaves': 5,
                'max_depth': 3,
                'min_data_in_leaf': 3,
                #'learning_rate': 0.01,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                #'lambda_l1': 0.1,
                #'lambda_l2': 0.1,
                #'min_gain_to_split': 0.1,
                'n_estimators': 10,
                'verbose': -1
            }
            bst = lgb.train(params, train_data, 10)
            y_pred = bst.predict(X_test, num_iteration=bst.best_iteration)
            y_pred = pd.Series(y_pred)
            y_pred.index = y_test.index
            mse = mean_squared_error(y_test, y_pred)
            #print("MSE:", mse)
            
            accuracy_df = pd.DataFrame(y_test)
            accuracy_df.columns = ["y_test"]
            accuracy_df["y_pred"] = y_pred
            accuracy_df["y_pred"] = accuracy_df["y_pred"].clip(0, 100)

            pred_today = []
            pred_today = bst.predict(last_row_test_X, num_iteration=bst.best_iteration)

            future_date = pd.to_datetime([last_row_date])
            pred_df = pd.DataFrame(pred_today, index=future_date, columns=[target])
            accuracy_df.rename(columns={'y_pred': target}, inplace=True)
            accuracy_df = accuracy_df.drop(columns=['y_test'])
            pred_df = pd.concat([pred_df, accuracy_df[~accuracy_df.index.isin(pred_df.index)]], axis=0)
            pred_df.rename(columns={target: "trend_pred"}, inplace=True)

            merged_df = pd.concat([df, pred_df], axis=1)
            merged_df["trend_pred"] = merged_df["trend_pred"].clip(0, 100)
            merged_df.rename(columns={"trend": "real", "trend_pred": "prediction"}, inplace=True)
            return merged_df, mse

        elif model != None:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred = np.round(y_pred).astype(int)

            y_pred = pd.Series(y_pred)
            y_pred.index = y_test.index
            mse = mean_squared_error(y_test, y_pred)
            #print("MSE:", mse)
            
            accuracy_df = pd.DataFrame(y_test)
            accuracy_df.columns = ["y_test"]
            accuracy_df["y_pred"] = y_pred
            accuracy_df["y_pred"] = accuracy_df["y_pred"].clip(0, 100)

            # Predict the next 7 days
            # Use the last 7 days of the dataset as features for prediction
            #last_row_test_X = X.tail(7)

            pred_today = model.predict(last_row_test_X)
            pred_today = np.round(pred_today).astype(int)

            # Create a DataFrame for the next 7 days predictions
            #future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=7, freq="D")
            future_date = pd.to_datetime([last_row_date])
            pred_df = pd.DataFrame(pred_today, index=future_date, columns=[target])
            accuracy_df.rename(columns={'y_pred': target}, inplace=True)
            accuracy_df = accuracy_df.drop(columns=['y_test'])
            pred_df = pd.concat([pred_df, accuracy_df[~accuracy_df.index.isin(pred_df.index)]], axis=0)
            pred_df.rename(columns={target: "trend_pred"}, inplace=True)

            #print(pred_df)


            # Pad the original data with 7 more days of empty data
            #padding_dates = future_dates
            #padding_df = pd.DataFrame(index=padding_dates, columns=df.columns)
            #padding_df.dropna(axis="columns", how="all", inplace=True)
            #df_padded = pd.concat([df, padding_df])

            #merged_df = df_padded[["trend"]].merge(pred_df[["trend_pred"]], left_index=True, right_index=True)
            #print(df)
            #print(pred_df)
            merged_df = pd.concat([df, pred_df], axis=1)
            merged_df["trend_pred"] = merged_df["trend_pred"].clip(0, 100)
            merged_df.rename(columns={"trend": "real", "trend_pred": "prediction"}, inplace=True)
            return merged_df, mse


    def run_all_models(self, df, corr_df):

        features = [
            #"views",
            #"likes",
            #"comments",
            #"days_old",
            "daily_views",
            "daily_likes",
            #"daily_comments",
            "daily_likes_to_views_ratio",
            #"daily_comments_to_views_ratio",
            #"trend_to_daily_views_ratio",
            #"trend_to_daily_likes_ratio",
            "diff_daily_views",
            "diff_daily_likes",
            #"diff_daily_comments",
            "day_number",
            "is_holiday"
        ]

        target = "trend"
        scores = {
            "lightgbm": 0,
            "randomforest": 0,
            "knn": 0,
            "linearregression": 0
        }
        lightgbm_result, scores["lightgbm"] = self.run_model("lightgbm", df, features, target)
        random_forest_result, scores["randomforest"] = self.run_model("randomforest", df, features, target)
        knn_result, scores["knn"] = self.run_model("knn", df, features, target)
        linear_regression_result, scores["linearregression"] = self.run_model("linearregression", df, features, target)

        return lightgbm_result, random_forest_result, knn_result, linear_regression_result, scores
    
