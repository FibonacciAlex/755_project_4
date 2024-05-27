import pandas as pd
import requests
from pytrends.request import TrendReq
from datetime import datetime, timedelta
import time
import os
from io import StringIO
import json


class FetchData:

    # Path & URL Strings:    
    def __init__(self, read_existing_files_only = False, fetch_first_item_only = False):
        self._news_items_json_path = "news_items.json"
        self._yt_api = "https://yt.lemnoslife.com/noKey/"
        self.read_existing_files_only = read_existing_files_only
        self._fetch_first_item_only = fetch_first_item_only
    
    @property
    def news_items_json_path(self):
        return self._news_items_json_path
    @news_items_json_path.setter
    def news_items_json_path(self, new_value):
        self._news_items_json_path = new_value
    @property
    def yt_api(self):
        return self._yt_api
    @yt_api.setter
    def yt_api(self, new_value):
        self._yt_api = new_value
    @property
    def read_existing_files_only(self):
        return self._read_existing_files_only
    @read_existing_files_only.setter
    def read_existing_files_only(self, new_value):
        if new_value == True or new_value == False:
            self._read_existing_files_only = new_value
        else:
            raise ValueError("'read_existing_files_only' property must be boolean!")
    @property
    def fetch_first_item_only(self):
        return self._fetch_first_item_only
    @fetch_first_item_only.setter
    def fetch_first_item_only(self, new_value):
        if new_value == True or new_value == False:
            self._fetch_first_item_only = new_value
        else:
            raise ValueError("'fetch_first_item_only' property must be boolean!")

    def read_news_items(self):
        path = self.news_items_json_path
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), path), "r") as file:
            data = file.read()
        df = pd.read_json(StringIO(data))
        df["news_items"] = df["news_items"].str.lower()
        if self.fetch_first_item_only == True:
            df = df.iloc[[0]]
        return df


    def fetch_trends_data(self, news_items: list):
        #pd.set_option("future.no_silent_downcasting", True) # Hides downcasting warning

        trends_df = pd.DataFrame()

        if self.read_existing_files_only == True:
            file_path = "./trends_data.csv"
            trends_df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path))
        else:
            pytrends = TrendReq()
            # Slice into N lists with maximum 5 items each:
            # This means each submitted keyword list will be max len 5, preventing "400" & "429" error codes from Google.

            for item in news_items:
                pytrends.build_payload([item], cat=0, timeframe="today 1-m")
                df = pytrends.interest_over_time()
                df = df.rename_axis("date").reset_index()
                df["date"] = pd.to_datetime(df["date"])
                df = df.drop(columns="isPartial")
                if len(trends_df) == 0:
                    trends_df = df
                else:
                    trends_df = pd.merge(trends_df, df, on="date", how="inner")

                # Sleep for 3s - try to avoid api rate limit:
                time.sleep(3)

            """
            sublists = [news_items[i:i + 5] for i in range(0, len(news_items), 5)]
            for list in sublists:
                pytrends.build_payload(list, cat=0, timeframe="today 1-m")
                df = pytrends.interest_over_time()
                df = df.rename_axis("date").reset_index()
                df["date"] = pd.to_datetime(df["date"])
                df = df.drop(columns="isPartial")
                if len(trends_df) == 0:
                    trends_df = df
                else:
                    trends_df = pd.merge(trends_df, df, on="date", how="inner")

                # Sleep for 6s - try to avoid api rate limit:
                time.sleep(3)
            """
            # Convert all columns to int except "date"
            num_columns = df.columns.difference(["date"])
            trends_df[num_columns] = df[num_columns].astype(int)
            trends_df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "trends_data.csv"), index=False)

        return trends_df


    def fetch_yt_videos_data(self, news__items: list, days_to_fetch = 30, read_from_file = False):
        def create_date_ranges(num_days):
            date_ranges = []
            today = datetime.now()
            
            for i in range(num_days):
                start_date = today - timedelta(days=i+1)
                end_date = start_date + timedelta(days=1)
                
                start_date_str = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
                end_date_str = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
                
                date_ranges.append({
                    "start_date_str": start_date_str,
                    "end_date_str": end_date_str
                })

            return date_ranges
        
        api = self.yt_api

        list_of_dfs = []

        if self.read_existing_files_only == False:
            # Fetch top X video for each day:
            endpoint = "search"
            max_results = 3

            # Get the timestamp for 1 month ago, formatted correctly:
            date_ranges = create_date_ranges(days_to_fetch)
            results = {}
            for item in news__items:
                print(f"Fetching video IDS for '{item}'.")
                videos = []
                for date in date_ranges:
                    params = {
                        "part": "snippet",
                        "maxResults": max_results,
                        "q": item,
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
                results[item] = videos
                print(f"Fetching video IDS for '{item}' is complete!.")
                time.sleep(0.06)
            
            # Get stats for each items" videos:
            final_results = {}
            for i, item in enumerate(results):
                print(f"Fetching video data for '{item}'.")
                endpoint = "videos"
                video_stats = []
                for video in results[item]:
                    params = {
                        "part": "statistics",
                        "id": video["id"]
                    }
                    response = requests.get(self.yt_api + endpoint, params=params)
                    data = response.json()
                    data["metadata"] = {"id": video["id"], "date": video["date"]}
                    video_stats.append(data)
                    print(f"Video data {i} for item '{item}' complete!.")
                final_results[item] = video_stats
            
            """
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "./item_videos.json"), "w") as json_file:
                json.dump(final_results, json_file, indent=4)
            """

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

            for i, item in enumerate(list_of_dfs):
                item.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"yt_data_{i}.csv"), index=False)
        else:
            for i, item in enumerate(news__items):
                file_path = f"./yt_data_{i}.csv"
                df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path))
                list_of_dfs.append(df)

        return list_of_dfs

    def merge_trends_and_ty_data(self, trends_df, yt_data_list):
        final_df_list = []

        for i, df in enumerate(yt_data_list):
            merged_df = df
            merged_df["trend"] = trends_df.iloc[:, i+1]
            final_df_list.append(merged_df)

        return final_df_list

    def fetch_and_return_final_df_list(self):
        news_items = self.read_news_items()
        trends = self.fetch_trends_data(news_items["news_items"].tolist())
        yt_data_list = self.fetch_yt_videos_data(news_items["news_items"].tolist())
        final_df_list = self.merge_trends_and_ty_data(trends, yt_data_list)
        return final_df_list

