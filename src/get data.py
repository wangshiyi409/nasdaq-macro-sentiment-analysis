import os
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ================================================================
#Consistent project root directory
# ================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # src/ directory
PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)  # go up to project root

RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

ensure_dir(RAW_DATA_DIR)

print("📁 Data will be saved to:", RAW_DATA_DIR)


# ================================================================
# 1. FRED DATA COLLECTOR
# ================================================================
class FREDDataCollector:
    def __init__(self, api_key):
        self.base_url = "https://api.stlouisfed.org/fred"
        self.api_key = api_key

    def get_series_data(self, series_id):
        """Fetch economic data series via FRED API"""
        url = f"{self.base_url}/series/observations"
        params = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json',
            'observation_start': '2014-12-31'
        }

        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data['observations'])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")[["value"]]
            return df.rename(columns={"value": series_id})
        else:
            print(f"❌ Error fetching data for {series_id}: {response.status_code}")
            return None


FRED_SERIES = {
    'GDP': 'GDP',
    'CPI': 'CPIAUCSL',
    'UNRATE': 'UNRATE',
    'FEDFUNDS': 'FEDFUNDS',
    'INDUSTRIAL_PRODUCTION': 'INDPRO',
    'RETAIL_SALES': 'RSAFS',
    'HOUSING_STARTS': 'HOUST',
    'DGS3MO': 'DGS3MO',
    'VIX': 'VIXCLS',
    'STLFSI': 'STLFSI',
    'TEDRATE': 'TEDRATE',
}


def fetch_fred_data(api_key):
    """Fetch all macroeconomic series from FRED"""
    collector = FREDDataCollector(api_key)
    dfs = []

    print("\n=== Fetching Macro + Financial Indicators from FRED ===")
    for name, series_id in FRED_SERIES.items():
        print(f"Fetching {name} ({series_id}) ...")
        df = collector.get_series_data(series_id)
        if df is not None:
            dfs.append(df)
        else:
            print(f"Skipping {series_id} due to error.")

    if dfs:
        df_macro = pd.concat(dfs, axis=1)
        output_path = os.path.join(RAW_DATA_DIR, "raw_US_macro_finance.xlsx")
        df_macro.to_excel(output_path)
        print(f"Saved: {output_path}")
        return df_macro
    else:
        print("❌ No macro data fetched.")
        return None


# ================================================================
# 2. FETCH WILSHIRE 5000
# ================================================================
def fetch_wilshire5000():
    print("\n=== Fetching WILSHIRE 5000 (^W5000) ===")
    w5000 = yf.download("^W5000", start="2014-12-31")
    w5000 = w5000.rename(columns={"Close": "WILSHIRE5000"})[["WILSHIRE5000"]]
    w5000.index = pd.to_datetime(w5000.index)

    output_path = os.path.join(RAW_DATA_DIR, "raw_w5000.xlsx")
    w5000.to_excel(output_path)
    print(f"Saved: {output_path}")
    return w5000


# ================================================================
# 3. FETCH NASDAQ INDEX (^IXIC)
# ================================================================
def fetch_nasdaq():
    print("\n=== Fetching NASDAQ Index (^IXIC) ===")
    nasdaq = yf.download("^IXIC", start="2014-12-31")
    nasdaq_close = nasdaq[["Close"]].rename(columns={"Close": "nasdaq_close"})

    output_path = os.path.join(RAW_DATA_DIR, "raw_nasdaq_close.xlsx")
    nasdaq_close.to_excel(output_path)
    print(f"Saved: {output_path}")
    return nasdaq_close


# ================================================================
# 4. SCRAPE NEWS FROM YAHOO FINANCE
# ================================================================
def scrape_yahoo_news_multiquery(queries):
    print("\n=== Scraping Yahoo Finance News (multi-query) ===")
    headers = {"User-Agent": "Mozilla/5.0"}
    all_news = []

    for query in queries:
        url = (
            "https://query1.finance.yahoo.com/v1/finance/search"
            f"?q={query}&newsCount=10&quotesCount=0&listsCount=0&offset=0"
        )

        resp = requests.get(url, headers=headers)
        data = resp.json()

        if "news" not in data:
            continue

        for item in data["news"]:
            all_news.append({
                "title": item.get("title"),
                "link": item.get("link"),
                "published_time": item.get("providerPublishTime"),
                "publisher": item.get("publisher"),
                "query": query,
                "source": "Yahoo_Search"
            })

    df = pd.DataFrame(all_news)
    output_path = os.path.join(RAW_DATA_DIR, "raw_yahoo_news.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    return df


# ================================================================
# MAIN EXECUTION
# ================================================================
if __name__ == "__main__":
    API_KEY = "dca474e4ebb8d4b9092e387e067c07cc"  #my API key

    # 1. Fetch FRED data
    fetch_fred_data(API_KEY)

    # 2. Fetch Wilshire 5000
    fetch_wilshire5000()

    # 3. Fetch NASDAQ Index
    fetch_nasdaq()

    # 4. Scrape Yahoo News (all nasdaq related)
    queries = ["nasdaq", "nasdaq futures", "nasdaq index", "stock market", "tech stocks"]
    scrape_yahoo_news_multiquery(queries)

    print("\n=== All data fetching complete. Files saved to data/raw ===")
