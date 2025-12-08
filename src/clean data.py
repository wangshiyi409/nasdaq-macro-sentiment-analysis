import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ================================================================
# Resolve project root & data paths
# ================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

ensure_dir(PROCESSED_DIR)

print(f"📁 Raw data directory: {RAW_DIR}")
print(f"📁 Processed data directory: {PROCESSED_DIR}")


# ================================================================
# Utility: Safe datetime conversion
# Remove non-date rows like "Ticker"
# ================================================================
def safe_datetime_index(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]   # remove invalid rows
    return df


# ================================================================
# 1️⃣ LOAD RAW DATA (READ FROM data/raw ONLY)
# ================================================================
def load_raw_data():
    macro = pd.read_excel(os.path.join(RAW_DIR, "raw_US_macro_finance.xlsx"), index_col=0)
    w5000 = pd.read_excel(os.path.join(RAW_DIR, "raw_w5000.xlsx"), index_col=0)
    nasdaq = pd.read_excel(os.path.join(RAW_DIR, "raw_nasdaq_close.xlsx"), index_col=0)
    news = pd.read_csv(os.path.join(RAW_DIR, "raw_yahoo_news.csv"))

    # Apply safe datetime parsing
    macro = safe_datetime_index(macro)
    w5000 = safe_datetime_index(w5000)
    nasdaq = safe_datetime_index(nasdaq)

    return macro, w5000, nasdaq, news


# ================================================================
# 2️⃣ CLEAN MACRO + MARKET DATA
# ================================================================
def process_macro_market_data(macro_finance_data, w5000, nasdaq_close):

    # Merge WILSHIRE5000
    df = macro_finance_data.join(w5000, how="outer")
    df["WILSHIRE5000"] = df["WILSHIRE5000"].ffill()

    # Buffett Indicator
    df["BUFFETT_INDICATOR"] = df["WILSHIRE5000"] / df["GDP"]

    # Merge NASDAQ close
    df = df.join(nasdaq_close, how="outer")
    df["nasdaq_close"] = df["nasdaq_close"].ffill()

    # Drop intermediate WILSHIRE col
    df = df.drop(columns=["WILSHIRE5000"], errors="ignore")

    # Delete columns with no values after 2025-01-01
    df_after = df.loc["2025-01-01":]
    valid_columns = df_after.notna().any(axis=0)
    final_cols = valid_columns[valid_columns].index.tolist()

    df_processed = df.loc[:, final_cols]

    # Export
    output_path = os.path.join(PROCESSED_DIR, "processed_market_data.xlsx")
    df_processed.to_excel(output_path)

    print(f"✅ Processed macro + market data saved to:\n{output_path}")

    return df_processed


# ================================================================
# 3️⃣ CLEAN YAHOO NEWS DATA
# ================================================================
def process_yahoo_news(df_raw):

    df = df_raw.copy()

    # Convert Unix timestamp → datetime
    df["published_datetime"] = pd.to_datetime(df["published_time"], unit="s")

    # Extract date
    df["published_date"] = df["published_datetime"].dt.date

    # Remove duplicate titles
    df = df.drop_duplicates(subset=["title"], keep="first")

    # Reorder columns
    df = df[[
        "published_date",
        "published_datetime",
        "title",
        "link",
        "publisher",
        "source"
    ]]

    df.index = range(df.shape[0])

    # Export
    output_path = os.path.join(PROCESSED_DIR, "processed_yahoo_news.xlsx")
    df.to_excel(output_path, index=False)

    print(f"✅ Processed Yahoo news saved to:\n{output_path}")

    return df


# ================================================================
# MAIN EXECUTION
# ================================================================
if __name__ == "__main__":
    print("\n=== Loading raw data ===")
    macro_raw, w5000_raw, nasdaq_raw, news_raw = load_raw_data()

    print("\n=== Cleaning macro + market data ===")
    processed_market = process_macro_market_data(macro_raw, w5000_raw, nasdaq_raw)

    print("\n=== Cleaning Yahoo news data ===")
    processed_news = process_yahoo_news(news_raw)

    print("\n🎉 All cleaned data saved in data/processed/")
