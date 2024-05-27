# Module Imports:
import pandas as pd
import requests
from pytrends.request import TrendReq
from datetime import datetime, timedelta
import time
import os
import json
import pprint # Pretty print
import statsmodels.formula.api as smf
import numpy as np
# from python_scripts.handle_model_util import handle_yt_date, handle_tr_data, train_model

# Path & URL Strings:
top_ten_json_path = "./top_ten.json"
yt_api = "https://yt.lemnoslife.com/noKey/"

only_run_first = True


#All models after training
trained_models = []


# Functions:

def read_top_ten(path):
    #df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), path), index_col="index")
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), path), "r") as file:
        data = file.read()
    df = pd.read_json(data)
    print(df)
    df["top_ten"] = df["top_ten"].str.lower()
    return df


def fetch_trends_data(top_ten_items: list):
    pd.set_option("future.no_silent_downcasting", True) # Hides downcasting warning
    pytrends = TrendReq()

    # Slice into N lists with maximum 5 items each:
    # This means each submitted keyword list will be max len 5, preventing "400" error codes from Google.
    sublists = [top_ten_items[i:i + 5] for i in range(0, len(top_ten_items), 5)]

    trends_df = pd.DataFrame()
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

        # Sleep for 60 ms - avoid api rate limit:
        time.sleep(0.06)
        if only_run_first:
            break;

    # Convert all columns to int except "date"
    int_columns = df.columns.difference(["date"])
    trends_df[int_columns] = df[int_columns].astype(int)
    trends_df.to_csv("trends.csv", index=False)
    return trends_df



def fetch_yt_videos_data(top_ten_items: list, api: str):
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

    # Fetch top 1 video for each day:
    endpoint = "search"
    max_results = 1

    # Get the timestamp for 1 month ago, formatted correctly:
    date_ranges = create_date_ranges(30)
    results = {}
    for item in top_ten_items:
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
            videos.append({
                "date": data["items"][0]["snippet"]["publishTime"],
                "id": data["items"][0]["id"]["videoId"]
            })
        results[item] = videos

        time.sleep(0.06)
        if only_run_first:
            break;
    
    # Get stats for each items" videos:
    final_results = {}
    for item in results:
        endpoint = "videos"
        video_stats = []
        for video in results[item]:
            params = {
                "part": "statistics",
                "id": video["id"]
            }
            response = requests.get(yt_api + endpoint, params=params)
            data = response.json()
            data["metadata"] = {"id": video["id"], "date": video["date"]}
            video_stats.append(data)
        final_results[item] = video_stats
    

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "./item_videos.json"), "w") as json_file:
        json.dump(final_results, json_file, indent=4)
    

    """
    global final_results
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "./item_videos.json"), "r") as file:
        final_results = json.load(file)
    """

    # Create the dataframe:
    list_of_dfs = []

    for entry in final_results:
        pprint.pprint(final_results[entry])
        data = []
        for i, val in enumerate(final_results[entry]):
            data.append({
                "date": val["metadata"]["date"],
                "id": val["metadata"]["id"],
                "views": int(val["items"][0]["statistics"]["viewCount"]) if val["items"][0]["statistics"].get("viewCount") is not None else 0,
                "likes": int(val["items"][0]["statistics"]["likeCount"]) if val["items"][0]["statistics"].get("likeCount") is not None else 0,
                "comments": int(val["items"][0]["statistics"]["commentCount"]) if val["items"][0]["statistics"].get("commentCount") is not None else 0,
                "favourites": int(val["items"][0]["statistics"]["favoriteCount"]) if val["items"][0]["statistics"].get("favoriteCount") is not None else 0
            })
        df = pd.DataFrame(data, index=None)
        print(df)
        list_of_dfs.append(df)

    for i, item in enumerate(list_of_dfs):
        item.to_csv(f"yt_data_{i}.csv", index=False)
    
    return list_of_dfs




# Main process:

try:
    # Read top ten from file:
    top_ten = read_top_ten(top_ten_json_path)
    print(top_ten)

    # Fetch trends - currently set to only fetch first news item! (palestine)
    trends = fetch_trends_data(top_ten["top_ten"].tolist())
    
    fetch_yt_videos_data(top_ten["top_ten"].tolist(), yt_api)
    # print(trends_data)

    #yt data should be fetch by API 
    # yt_data = fetch_yt_videos_data(top_ten['top_ten'].tolist())
    # trends_data = pd.read_csv('./python_scripts/trends.csv')
    # yt_data = []
    # yt_data[0] = pd.read_csv('./python_scripts/yt_data_0.csv')
    # print(yt_data)

    train_way = 1  #1= linear/poly linear    2=kNN  

    #handle yt data, for example, yt_data is a list, each one represents one lebel's data
    #but one label may have several videos in the same date   

    # for i, value in yt_data:
    #     formated_yt_data = handle_yt_date(yt_data)
    #     # pick up target trend data
    #     formated_tr_data = handle_tr_data(trends_data.iloc[:,[0,i]].copy())

    #     train_data = pd.merge(formated_yt_data, formated_tr_data,on='date', how='inner')
        
    #     trained_models[i] = train_model(train_data, train_way)
    


except Exception as e:
    print(f"ERROR: {e}")
    exit()


