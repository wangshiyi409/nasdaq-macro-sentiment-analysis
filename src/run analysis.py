import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, confusion_matrix

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ================================================================
# Resolve project root & data paths
# ================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

ensure_dir(PROCESSED_DIR)

print(f"📁 Processed data directory: {PROCESSED_DIR}")


# ================================================================
# 1️⃣ Load processed market data
# ================================================================
def load_market_data():
    path = os.path.join(PROCESSED_DIR, "processed_market_data.xlsx")
    df = pd.read_excel(path, index_col=0)
    df.index = pd.to_datetime(df.index)
    return df


# ================================================================
# 2️⃣ Correlation analysis
# ================================================================
def compute_correlations(market_data):
    macro_features = [
        'GDP', 'CPIAUCSL', 'UNRATE', 'FEDFUNDS', 'INDPRO',
        'RSAFS', 'HOUST', 'DGS3MO', 'VIXCLS', 'BUFFETT_INDICATOR'
    ]

    correlations = {}
    for feature in macro_features:
        temp = market_data[[feature, "nasdaq_close"]].dropna()
        correlations[feature] = temp[feature].corr(temp["nasdaq_close"])

    corr_df = (
        pd.DataFrame.from_dict(correlations, orient="index", columns=["correlation_with_nasdaq"])
        .sort_values("correlation_with_nasdaq", ascending=False)
    )

    output = os.path.join(PROCESSED_DIR, "processed_macro_correlation.xlsx")
    corr_df.to_excel(output)
    print(f"✅ Correlation saved → {output}")

    # strong features
    strong = corr_df[abs(corr_df["correlation_with_nasdaq"]) > 0.5].index.tolist()
    return strong, corr_df


# ================================================================
# 3️⃣ Tail-risk modeling (Logistic Regression)
# ================================================================
def tail_risk_modeling(market_data, strong_features):
    df = market_data.copy()
    df = df.sort_index()

    macro_cols = ["BUFFETT_INDICATOR", "RSAFS", "GDP", "CPIAUCSL", "HOUST", "DGS3MO", "FEDFUNDS"]
    df[macro_cols] = df[macro_cols].ffill().bfill()

    # 60-day forward drawdown
    horizon = 60
    df["min_price_60d"] = df["nasdaq_close"].rolling(window=horizon, min_periods=1).min().shift(-horizon + 1)
    df["min_price_60d"] = df["min_price_60d"].bfill()
    df["max_drawdown_60d"] = df["min_price_60d"] / df["nasdaq_close"] - 1
    df["max_drawdown_60d"] = df["max_drawdown_60d"].bfill()

    # classify tail risk
    risk_threshold = -0.02
    df["tail_risk_60d"] = (df["max_drawdown_60d"] <= risk_threshold).astype(int)

    df_model = df.dropna(subset=["tail_risk_60d"])
    X = df_model[macro_cols].ffill().bfill()
    y = df_model["tail_risk_60d"]

    # train-test split by date
    split_date = "2021-01-01"
    X_train = X[X.index < split_date]
    y_train = y[y.index < split_date]
    X_test = X[X.index >= split_date]
    y_test = y[y.index >= split_date]

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    proba_test = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba_test)

    print(f"\n📌 AUC for tail risk prediction: {auc:.3f}")
    print(classification_report(y_test, (proba_test > 0.5).astype(int)))

    # ROC data
    fpr, tpr, thresholds = roc_curve(y_test, proba_test)
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})
    auc_df = pd.DataFrame({"AUC": [auc]})

    # Confusion Matrix
    y_pred = (proba_test > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, columns=["Pred_0", "Pred_1"], index=["Actual_0", "Actual_1"])

    # export results
    output = os.path.join(PROCESSED_DIR, "processed_tail_risk_results.xlsx")
    with pd.ExcelWriter(output) as writer:
        roc_df.to_excel(writer, sheet_name="ROC_Data", index=False)
        auc_df.to_excel(writer, sheet_name="AUC", index=False)
        cm_df.to_excel(writer, sheet_name="Confusion_Matrix")

    print(f"✅ Tail-risk model results saved → {output}")


# ================================================================
# 4️⃣ Sentiment analysis
# ================================================================
def load_news_clean():
    path = os.path.join(PROCESSED_DIR, "processed_yahoo_news.xlsx")
    df = pd.read_excel(path)
    return df


def add_sentiment_scores(df):
    analyzer = SentimentIntensityAnalyzer()
    df = df.copy()
    df["sentiment"] = df["title"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
    return df


def sentiment_pipeline():
    df_clean = load_news_clean()
    df_sent = add_sentiment_scores(df_clean)

    output = os.path.join(PROCESSED_DIR, "processed_yahoo_news_sentiment.xlsx")
    df_sent.to_excel(output, index=False)

    sentiment_score = df_sent["sentiment"].mean()
    print(f"\n📌 Market Sentiment Score: {round(sentiment_score, 3)}")
    print(f"✅ Sentiment data saved → {output}")


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":

    print("\n=== Loading processed market data ===")
    market = load_market_data()

    print("\n=== Computing correlations ===")
    strong_features, correlation_table = compute_correlations(market)

    print("\n=== Running tail-risk modeling ===")
    tail_risk_modeling(market, strong_features)

    print("\n=== Running sentiment analysis ===")
    sentiment_pipeline()

    print("\n🎉 All analysis completed. Results saved in data/processed/")
