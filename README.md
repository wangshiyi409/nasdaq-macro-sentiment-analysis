# Macro Indicators & Web-Scraped Market Sentiment Analysis for NASDAQ Trends
DSCI 510 – Final Project  
Author: Shiyi Wang (shiyiw@usc.edu)  
USC ID: 9862305589
Github User Name: wangshiyi409

---

## 📌 Project Overview

This project investigates how **macro‐economic indicators** (GDP, CPI, Fed Funds Rate, Housing Starts, etc.) and **web‐scraped financial news sentiment** relate to trends in the **NASDAQ Composite Index**.

The full workflow includes:

1. **Data collection** (FRED API, Yahoo Finance API, Yahoo News Search)
2. **Data cleaning and merging**
3. **Correlation and tail‐risk modeling (Logistic Regression)**
4. **Sentiment scoring using VADER**
5. **Visualization of macro trends, ROC curve, sentiment charts, and more**

All scripts are located in the `src/` folder, and the project is fully reproducible.

---

## 📁 Repository Structure
```text
.
├── README.md
├── project_proposal.pdf
├── requirements.txt
│
├── data/
│   ├── raw/                # Raw macro data, NASDAQ data, web-scraped news
│   └── processed/          # Cleaned datasets, correlation tables, model outputs
│
├── results/
│   ├── Final Report.pdf          
│   ├── correlation_heatmap.png
│   ├── macro_vs_nasdaq.png
│   ├── roc_curve.png
│   ├── confusion_matrix.png
│   ├── sentiment_score.png
│   └── sentiment_pie_wordcloud.png
│
└── src/
    ├── get_data.py         # Fetch FRED API data, Yahoo Finance, Yahoo News
    ├── clean_data.py       # Clean and preprocess datasets
    ├── run_analysis.py     # Correlation, tail-risk modeling, sentiment scoring
    └── visualize_results.py# Generate all project plots
```
---

## ⚙️ Installation Instructions

### 1️⃣ Create and activate a virtual environment

python -m venv venv
source venv/bin/activate # Mac/Linux
venv\Scripts\activate # Windows

### 2️⃣ Install all required dependencies

pip install -r requirements.txt


The requirements include:
- pandas  
- numpy
- beautifulsoup4
- matplotlib  
- seaborn  
- scikit-learn  
- requests  
- yfinance
- vaderSentiment
- wordcloud
- openpyxl
- textwrap

---

## 📥 Step 1 — Data Collection

Fetch macroeconomic data (FRED), NASDAQ data, Wilshire 5000, and news headlines.

Run:
python src/get_data.py

Outputs saved to:
data/raw/

---

## 🧹 Step 2 — Data Cleaning & Processing

This step merges macro indicators, computes Buffett indicator, cleans news data, and prepares modeling datasets.

Run:
python src/clean_data.py

Outputs saved to:
data/processed/

---

## 📊 Step 3 — Analysis (Correlation, Modeling, Sentiment Scoring)

This script performs:

- Computes macro–NASDAQ correlations
- Builds a logistic regression model to classify NASDAQ 60-day tail-risk
- Generates ROC curve data & confusion matrix
- Computes average news sentiment score
- Saves processed analysis tables

Run:
python src/run_analysis.py

Outputs saved to:
data/processed/

---

## 📈 Step 4 — Visualization

Generate all plots used in the final report:

- Macro–NASDAQ 8-panel time-series
- Correlation heatmap
- ROC Curve
- Confusion Matrix
- Sentiment score bar (red→gray→blue gradient)
- Sentiment pie chart
- WordCloud of news headlines

Run:
python src/visualize_results.py

Outputs saved to:
results/

---

## 📝 Final Report

The **final_report.pdf** summarizes:

- Motivation and research question  
- Data collection and API sources  
- Cleaning and analysis methods  
- Visualizations and interpretation  
- Changes from original proposal  
- Future work  

This file is located under:
results/final_report.pdf

---

## 🚀 How to Reproduce the Entire Pipeline

To reproduce the full workflow from raw data to final figures:

- python src/get_data.py
- python src/clean_data.py
- python src/run_analysis.py
- python src/visualize_results.py

---

## ✔️ Notes

- All scripts use relative project paths, so they work on any machine.
- If a FRED API key becomes invalid, update it in src/get_data.py.

---

