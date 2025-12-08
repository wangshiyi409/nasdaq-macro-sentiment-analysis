import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from matplotlib.colors import LinearSegmentedColormap


# ================================================================
# Resolve project paths
# ================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

ensure_dir(RESULTS_DIR)

print(f"📁 Processed: {PROCESSED_DIR}")
print(f"📁 Results:   {RESULTS_DIR}")


# ================================================================
# Global Theme Colors 
# ================================================================
COLOR_NEG = "#e74c3c"   # Red
COLOR_NEU = "#bdc3c7"   # Gray
COLOR_POS = "#3498db"   # Blue

COLOR_MAIN = COLOR_POS  # main variables：blue
COLOR_REF  = COLOR_NEG  # NASDAQ：red

COLOR_PALETTE = [COLOR_NEG, COLOR_NEU, COLOR_POS]


# ================================================================
# Load processed data
# ================================================================
def load_data():
    df_corr = pd.read_excel(os.path.join(PROCESSED_DIR, "processed_macro_correlation.xlsx"))
    df_market = pd.read_excel(os.path.join(PROCESSED_DIR, "processed_market_data.xlsx"), index_col=0)
    df_market.index = pd.to_datetime(df_market.index)

    roc_df = pd.read_excel(os.path.join(PROCESSED_DIR, "processed_tail_risk_results.xlsx"), sheet_name="ROC_Data")
    auc_df = pd.read_excel(os.path.join(PROCESSED_DIR, "processed_tail_risk_results.xlsx"), sheet_name="AUC")
    cm_df  = pd.read_excel(os.path.join(PROCESSED_DIR, "processed_tail_risk_results.xlsx"), sheet_name="Confusion_Matrix", index_col=0)

    df_sent = pd.read_excel(os.path.join(PROCESSED_DIR, "processed_yahoo_news_sentiment.xlsx"))

    return df_corr, df_market, roc_df, auc_df, cm_df, df_sent


# ================================================================
# 1️⃣ Correlation Heatmap（keep coolwarm）
# ================================================================
def plot_heatmap(df_corr):
    if "Indicator" not in df_corr.columns:
        df_corr = df_corr.rename(columns={"Unnamed: 0": "Indicator"})

    df = df_corr.set_index("Indicator").T

    plt.figure(figsize=(14, 3))
    sns.heatmap(
        df, annot=True, cmap="coolwarm",
        fmt='.3f', center=0, linewidths=0.5, linecolor="white"
    )
    plt.title("Economic Indicators Correlation Heatmap", fontsize=16)
    plt.xticks(rotation=45, ha="right")

    path = os.path.join(RESULTS_DIR, "correlation_heatmap.png")
    plt.savefig(path)
    plt.close()
    print(f"✅ Saved: {path}")


# ================================================================
# 2️⃣ Macro vs NASDAQ (4×2 plots) 
# ================================================================
def plot_macro_vs_nasdaq(df_market):
    features = ['BUFFETT_INDICATOR','RSAFS','GDP','CPIAUCSL','HOUST','DGS3MO','FEDFUNDS']
    df_plot = df_market[features + ["nasdaq_close"]].dropna(how="all")

    sns.set_style("whitegrid")

    fig, axes = plt.subplots(4, 2, figsize=(17, 18))
    axes = axes.flatten()

    for ax, feature in zip(axes, features):

        # Macro blue
        df_plot[feature].dropna().plot(ax=ax, color=COLOR_MAIN, linewidth=2, label=feature)

        # NASDAQ red
        ax2 = ax.twinx()
        df_plot["nasdaq_close"].dropna().plot(ax=ax2, color=COLOR_REF, alpha=0.65, linewidth=2, label="NASDAQ")

        ax.set_title(f"{feature} vs NASDAQ", fontsize=13, fontweight="bold")
        ax.legend(loc="upper left", frameon=False)
        ax2.legend(loc="upper right", frameon=False)

        ax.grid(axis="x", linestyle="--", alpha=0.5)
        ax2.grid(axis="y", linestyle="--", alpha=0.4)

    for ax in axes[len(features):]:
        ax.set_visible(False)

    fig.suptitle("Macro Indicators vs NASDAQ Composite Index", fontsize=18, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, "macro_vs_nasdaq.png")
    plt.savefig(path)
    plt.close()
    print(f"✅ Saved: {path}")


# ================================================================
# 3️⃣ ROC Curve
# ================================================================
def plot_roc(roc_df, auc_df):
    auc = auc_df["AUC"].iloc[0]

    plt.figure(figsize=(8, 6))
    plt.plot(roc_df["fpr"], roc_df["tpr"], color=COLOR_POS, linewidth=2, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")

    plt.title("ROC Curve", fontsize=14, fontweight="bold")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(alpha=0.6)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, "roc_curve.png")
    plt.savefig(path)
    plt.close()
    print(f"✅ Saved: {path}")


# ================================================================
# 4️⃣ Confusion Matrix
# ================================================================
def plot_confusion_matrix(cm_df):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(path)
    plt.close()
    print(f"✅ Saved: {path}")


# ================================================================
# 5️⃣ Sentiment Score（Red → Gray → Blue gradient）
# ================================================================
def plot_sentiment_score(df_sent):
    score = df_sent["sentiment"].mean()

    cmap = LinearSegmentedColormap.from_list("sentiment_cmap", COLOR_PALETTE)
    gradient = np.linspace(0, 1, 500).reshape(1, -1)

    plt.figure(figsize=(10, 2))
    plt.imshow(gradient, aspect="auto", cmap=cmap, extent=[-1, 1, 0.9, 1.1])

    plt.scatter(score, 1, color="red", s=160)

    plt.xlim(-1, 1)
    plt.yticks([])
    plt.xlabel("Negative ←———— Market Sentiment ————→ Positive")
    plt.title(f"Market Sentiment Score = {score:.3f}")

    path = os.path.join(RESULTS_DIR, "sentiment_score.png")
    plt.savefig(path)
    plt.close()
    print(f"✅ Saved: {path}")


# ================================================================
# 6️⃣ Pie Chart + WordCloud（color matching united / coolwarm）
# ================================================================
def plot_sentiment_pie_wordcloud(df_sent):

    def classify_sent(score):
        if score > 0.2: return "Positive"
        if score < -0.2: return "Negative"
        return "Neutral"

    df_sent["class"] = df_sent["sentiment"].apply(classify_sent)

    color_map = {
        "Positive": COLOR_POS,
        "Neutral":  COLOR_NEU,
        "Negative": COLOR_NEG
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ------- Pie Chart -------
    df_sent["class"].value_counts().plot(
        kind="pie",
        colors=[color_map[c] for c in df_sent["class"].value_counts().index],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"linewidth":1, "edgecolor":"white"},
        ax=axes[0]
    )
    axes[0].set_title("Sentiment Composition")
    axes[0].set_ylabel("")

    # ------- WordCloud（coolwarm red→blue） -------
    text = " ".join(df_sent["title"].astype(str).tolist())
    wc = WordCloud(
        width=600, height=400,
        background_color="white",
        colormap="coolwarm"     # red→blue 
    ).generate(text)

    axes[1].imshow(wc, interpolation="bilinear")
    axes[1].axis("off")
    axes[1].set_title("Keyword WordCloud")

    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, "sentiment_pie_wordcloud.png")
    plt.savefig(path)
    plt.close()
    print(f"✅ Saved: {path}")


# ================================================================
# MAIN ENTRY
# ================================================================
if __name__ == "__main__":
    df_corr, df_market, roc_df, auc_df, cm_df, df_sent = load_data()

    print("\n=== Generating visualizations with unified color theme ===")
    plot_heatmap(df_corr)
    plot_macro_vs_nasdaq(df_market)
    plot_roc(roc_df, auc_df)
    plot_confusion_matrix(cm_df)
    plot_sentiment_score(df_sent)
    plot_sentiment_pie_wordcloud(df_sent)

    print("\n🎉 All visualizations saved to /results/ !")

