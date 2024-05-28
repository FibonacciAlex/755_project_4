# Module Imports:
import pandas as pd
from datetime import datetime
import statsmodels.formula.api as smf
import numpy as np



def create_random_yt_data(date_range):
    np.random.seed(0) 
    views = np.random.randint(0, 30001, size=len(date_range))
    likes = np.random.randint(0, 30001, size=len(date_range))
    comments = np.random.randint(0, 30001, size=len(date_range))

    df = pd.DataFrame({
        'date': date_range,
        'views': views,
        'likes': likes,
        'comments': comments
    })

def formate_linear_formula(features, predict_feat): 
    result = '+'.join(features)
    result = predict_feat + '~' + result
    return result


def train_linear_model(data, formula_str):
#     multi_linear = smf.ols(formula='views ~ likes + comments + palistine', data=data).fit()
    multi_linear = smf.ols(formula=formula_str, data=data).fit()
    
    multi_linear.summary() #this step can show the OLS Regression Results
    return multi_linear

def formate_date(date_str):
    date_obj = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%SZ')
    new_date_str = date_obj.strftime('%Y-%m-%d')
    return new_date_str

#In this method, we should combine or aggregate by date and get mean value
def handle_yt_data(df:pd.DataFrame):
    #formate date column and remove time value
    
    df['date'] = pd.to_datetime(df['date']).dt.date

    # 2. groupby date and get mean value
    grouped = df.groupby('date').agg(
        view_mean=('views', 'mean'),
        like_mean=('likes', 'mean'),
        com_mean=('comments', 'mean'),
        fav_mean=('favourites', 'mean')
    ).reset_index()

    # 3. calculate per day value
    current_date = pd.to_datetime('today').date()
    grouped['days_diff'] = (current_date - grouped['date']).apply(lambda x: x.days)

    grouped['view_per_day'] = grouped['view_mean'] / grouped['days_diff']
    grouped['like_per_day'] = grouped['like_mean'] / grouped['days_diff']
    grouped['com_per_day'] = grouped['com_mean'] / grouped['days_diff']
    grouped['fav_per_day'] = grouped['fav_mean'] / grouped['days_diff']

    # print(grouped[['date', 'view_per_day', 'like_per_day', 'com_per_day', 'fav_per_day']])
    # print(grouped)
    
    return grouped

def train_model(df:pd.DataFrame, train_way):
    model = None
    if train_way ==1:
        model = train_linear_model(df, 'trends~view_per_day+like_per_day+com_per_day')
    return model


def normalize_trend(df, col_index):
    cols = df.columns
    col_name = cols[col_index]
    min_val = df[col_name].min()
    max_val = df[col_name].max()
    df[col_name] = (df[col_name] - min_val) / (max_val - min_val)
    return df


def handle_tr_data(df:pd.DataFrame):    
    df['date'] = pd.to_datetime(df['date']).dt.date
    df.rename(columns={df.columns[1]: 'trend'}, inplace=True)
    # normalize 
    df = normalize_trend(df, 1)
    return df