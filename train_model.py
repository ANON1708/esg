import pandas as pd
import numpy as np
from transformers import pipeline
from newsapi import NewsApiClient
from fuzzywuzzy import fuzz
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
import os

# Limit threads to prevent crash
os.environ["OMP_NUM_THREADS"] = "1"

# Load ESG dataset
df = pd.read_csv("SP 500 ESG Risk Ratings.csv")

# Drop rows with missing key scores
df = df.dropna(subset=["Total ESG Risk score", "Environment Risk Score", "Governance Risk Score", "Social Risk Score"])

# Fill missing values
df.fillna({
    "Controversy Score": df["Controversy Score"].median(),
    "Sector": "Unknown",
    "Industry": "Unknown"
}, inplace=True)

# Clean ESG Risk Percentile
df["ESG Risk Percentile"] = df["ESG Risk Percentile"].astype(str).str.extract(r'(\d+)').astype(float)

# Normalize company names
df["Clean Name"] = df["Name"].str.lower().str.replace(r"[^\w\s]", "", regex=True)

# Load FinBERT pipeline
finbert = pipeline("text-classification", model="ProsusAI/finbert")

# NewsAPI setup
newsapi = NewsApiClient(api_key="fcda71eae5464edaa75e8b5839ac30cb")  # Replace or load from secrets

# Fetch ESG-related headlines
news_headlines = []
for page in range(1, 4):
    articles = newsapi.get_everything(q='ESG sustainability', language='en', sort_by='publishedAt', page_size=20, page=page)
    news_headlines += [article['title'] for article in articles['articles']]

# Run FinBERT on headlines
news_sentiments = [finbert(headline) for headline in news_headlines]

# Match sentiments to companies
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

print(f"✅ Sentiment coverage: {matched_count / len(df):.0%}")

# Define features + target
features = ["Environment Risk Score", "Governance Risk Score", "Social Risk Score", "Controversy Score", "ESG Risk Percentile", "News Sentiment Score"]
X = df[features]
y = df["Total ESG Risk score"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Predict & evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Evaluation:")
print(f"✅ Mean Absolute Error (MAE): {mae:.2f}")
print(f"✅ Mean Squared Error (MSE): {mse:.2f}")
print(f"✅ R² Score: {r2:.2f}")

# Save the trained model
joblib.dump(model, "esg_model.pkl")
print("✅ Model saved as 'esg_model.pkl'")

