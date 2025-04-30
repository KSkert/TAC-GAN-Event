import pandas as pd
import time
import random
from datetime import datetime, timedelta
from pytrends.request import TrendReq
from tqdm import tqdm

df = pd.read_csv("top_wikipedia_event_entertainment_pages_from_2015.csv")
df["date"] = pd.to_datetime(df["date"])

# basic approach
# pytrends = TrendReq(hl="en-US", tz=360)
# pytrends retry logic to avoid bot block
pytrends = TrendReq(hl='en-US', tz=360, timeout=(10,25), proxies=['https://34.203.233.13:80',], retries=2, backoff_factor=0.1, requests_args={'verify':False})

def fetch_google_trends(keyword, start_date):
    """
    Fetch Google Trends daily interest data for a keyword.
    - One month before start_date (-30 days).
    - Three months after start_date (+90 days).
    Returns two lists: one for the prior month and one for the following three months.
    """
    start_date_prior = (start_date - timedelta(days=30)).strftime("%Y-%m-%d")  # 1 month before event date >>> changeable
    end_date_post = (start_date + timedelta(days=90)).strftime("%Y-%m-%d")    # 3 months after event date >>> changeable
    time_range = f"{start_date_prior} {end_date_post}"
    
    try:
        pytrends.build_payload([keyword], cat=0, timeframe=time_range, geo="", gprop="")
        trends_data = pytrends.interest_over_time()

        if trends_data.empty:
            print(f" Nothing for: {keyword}")
            return None, None
        
        trends_data = trends_data.reset_index()

        # Ensure 'value' column exists
        if "value" not in trends_data.columns:
            print(f"No 'value' column for {keyword}. Skipping...")
            return None, None
        
        # extract values separately
        prior_month_data = trends_data[trends_data['date'] < start_date].get("value", []).tolist()
        post_three_months_data = trends_data[trends_data['date'] >= start_date].get("value", []).tolist()

        if prior_month_data is None or post_three_months_data is None:
            print(f"Skipping {keyword} because we didn't find any trends data ")
            return None, None
        
        return prior_month_data, post_three_months_data
    
    except Exception as e:
        print(f"Sudden error fetching {keyword}: {e}")
        return None, None

# Store results
new_data = []

# tqdm for progress tracking
for _, row in tqdm(df.iterrows(), total=len(df), desc="Fetching Trends"):
    event_title = row["article_title"]
    event_date = row["date"]
    
    trend_list_prior, trend_list_post = fetch_google_trends(event_title, event_date)
    
    if trend_list_prior is not None and trend_list_post is not None:
        new_data.append([row["date"], event_title, trend_list_prior, trend_list_post])
    
    time.sleep(2)  # pause to avoid rate limiting 

final_df = pd.DataFrame(new_data, columns=["date", "article_title", "trend_list_month_prior", "trend_list_3months_post"])
final_df.to_csv("google_trends_data.csv", index=False)
print("Google Trends data saved successfully.")