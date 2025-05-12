# train_model.py — Updated to create merged dataset first

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
from transformers import pipeline
from newsapi import NewsApiClient
import os

MODEL_PATH = os.getenv("MODEL_PATH", "esg_model.pkl")
DATA_PATH = os.getenv("DATA_PATH", "merged_esg_data.csv")
HISTORY_FILE = os.getenv("HISTORY_FILE", "esg_score_history.csv")
DELTA_LOG_FILE = os.getenv("DELTA_LOG_FILE", "delta_log.csv")


# === Load and Combine Datasets ===
print("🔄 Loading datasets...")

# Load existing S&P 500 dataset
df_sp500 = pd.read_csv("SP 500 ESG Risk Ratings.csv")

# Load additional ESG dataset
df_extra = pd.read_csv("data.csv")

# Standardize column names for extra dataset
cols = {
    "ticker": "Symbol",
    "name": "Name",
    "environment_score": "Environment Risk Score",
    "social_score": "Social Risk Score",
    "governance_score": "Governance Risk Score",
    "total_score": "Total ESG Risk score"
}
df_extra = df_extra.rename(columns=cols)
df_extra["Controversy Score"] = 0
df_extra["ESG Risk Percentile"] = 50  # default fallback if missing

required_cols = ["Symbol", "Name", "Environment Risk Score", "Social Risk Score", "Governance Risk Score",
                 "Controversy Score", "ESG Risk Percentile", "Total ESG Risk score"]

# Combine the two datasets
df_combined = pd.concat([df_sp500[required_cols], df_extra[required_cols]], ignore_index=True)

print(f"✅ Combined dataset shape: {df_combined.shape}")

# === Save merged dataset ===
df_combined.to_csv("merged_esg_data.csv", index=False)
print("✅ Merged dataset saved as 'merged_esg_data.csv'")

# === Clean & Initialize ===
df_combined = df_combined.dropna(subset=["Environment Risk Score", "Social Risk Score", "Governance Risk Score", "Total ESG Risk score"])
df_combined["ESG Risk Percentile"] = (
    df_combined["ESG Risk Percentile"].astype(str).str.extract(r'(\d+)').astype(float).fillna(50.0)
)
df_combined["News Sentiment Score"] = 0.0

# === Load FinBERT + NewsAPI ===
print("🚀 Fetching FinBERT sentiment for training data...")
finbert = pipeline("text-classification", model="ProsusAI/finbert")
newsapi = NewsApiClient(api_key=os.getenv("NEWS_API_KEY", ""))

df_combined["Clean Name"] = df_combined["Name"].str.lower().str.replace(r'[^\w\s]', '', regex=True)

for idx, row in df_combined.iterrows():
    name = row["Clean Name"]
    try:
        articles = newsapi.get_everything(q=name, language='en', sort_by='relevancy', page_size=5)
        headlines = [a['title'] for a in articles['articles'] if a['title']]
        sentiments = [finbert(h)[0]['label'] for h in headlines]
        if sentiments:
            avg = sum({"positive": 1, "neutral": 0, "negative": -1}[s] for s in sentiments) / len(sentiments)
            df_combined.at[idx, "News Sentiment Score"] = avg
    except:
        continue

# === Train-Test Split ===
features = ["Environment Risk Score", "Governance Risk Score", "Social Risk Score",
            "Controversy Score", "ESG Risk Percentile", "News Sentiment Score"]
target = "Total ESG Risk score"

X = df_combined[features]
y = df_combined[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"📊 Training Data: {X_train.shape}, Testing Data: {X_test.shape}")

# === Train XGBoost ===
xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror", n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

print("🚀 Training model...")
xgb_model.fit(X_train, y_train)

# === Evaluate ===
y_pred = xgb_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Evaluation:")
print(f"✅ MAE: {mae:.2f}")
print(f"✅ MSE: {mse:.2f}")
print(f"✅ R^2: {r2:.2f}")

# === Save model ===
joblib.dump(xgb_model, "esg_model.pkl")
print("✅ Model saved as 'esg_model.pkl'")
