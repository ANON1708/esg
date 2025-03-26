import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline
from newsapi import NewsApiClient
from fuzzywuzzy import fuzz
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Limit threads to prevent crashes
os.environ["OMP_NUM_THREADS"] = "1"

st.title("⚡ Realtime ESG Risk Score Model")
st.markdown("This app predicts real-time ESG risk scores based on company ESG data and ESG-related news sentiment using a pretrained XGBoost model.")

# Load pretrained XGBoost model
model = joblib.load("esg_model.pkl")

# Load default S&P 500 ESG data
df = pd.read_csv("SP 500 ESG Risk Ratings.csv")
st.subheader("📂 Preview of Default ESG Dataset")
st.dataframe(df.head())

# FinBERT pipeline
finbert = pipeline("text-classification", model="ProsusAI/finbert")

# NewsAPI setup
newsapi = NewsApiClient(api_key=st.secrets["NEWS_API_KEY"])

# Fetch ESG news
st.info("Fetching ESG-related news articles from the past few days...")
news_headlines = []
for page in range(1, 4):
    articles = newsapi.get_everything(q='ESG sustainability', language='en', sort_by='publishedAt', page_size=20, page=page)
    news_headlines += [article['title'] for article in articles['articles']]

st.success(f"Fetched {len(news_headlines)} news articles (Timeframe: Past ~3 days via NewsAPI)")

# Run FinBERT on headlines
news_sentiments = [finbert(headline) for headline in news_headlines]

# Clean and prepare data
df = df.dropna(subset=["Environment Risk Score", "Governance Risk Score", "Social Risk Score"])
df.fillna({
    "Controversy Score": df["Controversy Score"].median(),
    "Sector": "Unknown",
    "Industry": "Unknown"
}, inplace=True)
df["ESG Risk Percentile"] = df["ESG Risk Percentile"].astype(str).str.extract(r'(\d+)').astype(float)
df["Clean Name"] = df["Name"].str.lower().str.replace(r'[^\w\s]', '', regex=True)

# Match sentiment to companies
df["News Sentiment"] = "neutral"
df["News Sentiment Score"] = 0.0
company_sentiments = {}
for i, headline in enumerate(news_headlines):
    headline_lower = headline.lower()
    for idx, row in df.iterrows():
        score = fuzz.partial_ratio(row["Clean Name"], headline_lower)
        if score >= 60:
            symbol = row["Symbol"]
            if symbol not in company_sentiments:
                company_sentiments[symbol] = []
            company_sentiments[symbol].append(news_sentiments[i][0]["label"])

matched_count = 0
for idx, row in df.iterrows():
    symbol = row["Symbol"]
    if symbol in company_sentiments:
        labels = company_sentiments[symbol]
        avg_score = sum({"positive": 1, "neutral": 0, "negative": -1}[l] for l in labels) / len(labels)
        df.at[idx, "News Sentiment Score"] = avg_score
        df.at[idx, "News Sentiment"] = "positive" if avg_score > 0.25 else "negative" if avg_score < -0.25 else "neutral"
        matched_count += 1

st.write(f"✅ Sentiment coverage: {matched_count / len(df):.0%}")

# Prediction only (no retraining)
features = ["Environment Risk Score", "Governance Risk Score", "Social Risk Score", "Controversy Score", "ESG Risk Percentile", "News Sentiment Score"]
X = df[features]
df["Predicted ESG Risk Score"] = model.predict(X)

# Add Risk Level interpretation
def interpret_risk(score):
    if score < 10:
        return "Negligible"
    elif score < 20:
        return "Low"
    elif score < 30:
        return "Medium"
    elif score < 40:
        return "High"
    else:
        return "Severe"

df["Risk Level"] = df["Predicted ESG Risk Score"].apply(interpret_risk)
df["Risk Score Delta"] = df["Predicted ESG Risk Score"] - df["Total ESG Risk score"]

# Choose which score to view
st.subheader("📊 Choose Score Type to Display")
score_option = st.selectbox("Select ESG Score Type", ["Predicted ESG Risk Score", "Total ESG Risk score"])

# User search for specific company
st.subheader("🔍 Search ESG Score for a Company")
search_name = st.text_input("Enter Company Name or Symbol to check ESG Risk")
if search_name:
    search_name_clean = search_name.lower().strip()
    matches = df[df["Name"].str.lower().str.contains(search_name_clean) | df["Symbol"].str.lower().str.contains(search_name_clean)]
    if not matches.empty:
        st.write("### ESG Risk for Matching Companies")
        st.dataframe(matches[["Symbol", "Name", score_option, "Risk Level", "Risk Score Delta"]])
    else:
        st.warning("No matching companies found in the dataset.")

# Search custom company with manual entry
df_input = st.text_input("🔍 Predict ESG score for any company (type name)")
if df_input:
    with st.spinner("Predicting ESG Risk using sentiment for: " + df_input):
        sentiment_result = finbert(df_input)
        label = sentiment_result[0]['label']
        score = {"positive": 1, "neutral": 0, "negative": -1}[label]
        st.success(f"Sentiment for '{df_input}' is {label} ({score}) — based on headlines fetched in the past ~3 days")
        st.write("This is a basic text-level estimation. For full ESG prediction, please match to company profiles.")

# Show top 10 risky companies
st.subheader("🔍 Top 10 Companies by Predicted ESG Risk")
st.dataframe(df[["Symbol", "Name", "Predicted ESG Risk Score", "Total ESG Risk score", "Risk Level", "Risk Score Delta"]]
             .sort_values("Predicted ESG Risk Score", ascending=False)
             .head(10))

# 📊 Visualization: Distribution of Predicted Risk Scores
st.subheader("📈 ESG Risk Score Distribution")
fig, ax = plt.subplots()
sns.histplot(df["Predicted ESG Risk Score"], kde=True, bins=30, ax=ax)
ax.set_xlabel("Predicted ESG Risk Score")
ax.set_ylabel("Count")
st.pyplot(fig)

# 📊 Visualization: Risk Levels
st.subheader("📊 Risk Level Breakdown")
risk_counts = df["Risk Level"].value_counts().sort_index()
st.bar_chart(risk_counts)

# 💾 Download full results
csv = df.to_csv(index=False)
st.download_button("📥 Download Full ESG Risk Predictions", data=csv, file_name="esg_predictions.csv", mime="text/csv")
