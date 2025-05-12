# esg_model.py — ESG Risk Monitoring from Merged Dataset + Gradio App

import pandas as pd
import numpy as np
import joblib
import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from transformers import pipeline
from newsapi import NewsApiClient
from fuzzywuzzy import fuzz
import os
import sys
import gradio as gr

MODEL_PATH = os.getenv("MODEL_PATH", "esg_model.pkl")
DATA_PATH = os.getenv("DATA_PATH", "merged_esg_data.csv")
HISTORY_FILE = os.getenv("HISTORY_FILE", "esg_score_history.csv")
DELTA_LOG_FILE = os.getenv("DELTA_LOG_FILE", "delta_log.csv")

# === Settings ===
ALERT_THRESHOLD = 10.0  # Risk difference needed to trigger alert
TOP_N = 3  # Companies to include in alert

# === Load FinBERT sentiment model ===
finbert = pipeline("text-classification", model="ProsusAI/finbert")

# === Load trained ESG Risk model ===
model = joblib.load(MODEL_PATH)

# === Load merged ESG dataset ===
print("🔄 Loading merged ESG dataset...")
df = pd.read_csv(DATA_PATH)

# === Clean and prepare ===
df = df.dropna(subset=["Environment Risk Score", "Governance Risk Score", "Social Risk Score", "Total ESG Risk score"])
df["Controversy Score"] = df.get("Controversy Score", 0)
df["ESG Risk Percentile"] = (
    df["ESG Risk Percentile"].astype(str).str.extract(r'(\d+)').astype(float).fillna(50.0)
)
df["Clean Name"] = df["Name"].str.lower().str.replace(r'[\W_]+', ' ', regex=True)

if len(sys.argv) == 1 or sys.argv[1] != "gradio_ui":
    # === Generate fresh sentiment scores using FinBERT ===
    print("🚀 Generating sentiment scores using FinBERT...")
    newsapi = NewsApiClient(api_key=os.getenv("NEWS_API_KEY", ""))
    df["News Sentiment Score"] = 0.0
    sentiment_mapping = {"positive": 1, "neutral": 0, "negative": -1}

    for idx, row in df.iterrows():
        name = row["Clean Name"]
        try:
            articles = newsapi.get_everything(q=name, language='en', sort_by='relevancy', page_size=5)
            headlines = [article['title'] for article in articles['articles'] if article['title']]
            sentiments = [finbert(h)[0]['label'].lower() for h in headlines]
            if sentiments:
                avg = sum(sentiment_mapping.get(s, 0) for s in sentiments) / len(sentiments)
                df.at[idx, "News Sentiment Score"] = avg
        except Exception as e:
            print(f"⚠️ Skipping {name}: {e}")
            continue

    # === Predict ESG Risk ===
    features = ["Environment Risk Score", "Governance Risk Score", "Social Risk Score",
                "Controversy Score", "ESG Risk Percentile", "News Sentiment Score"]
    X = df[features]

    print("🚀 Predicting updated ESG Risk Scores...")
    df["Predicted ESG Risk Score"] = model.predict(X)

    # === Add timestamp for tracking ===
    timestamp = datetime.datetime.now()
    df["Timestamp"] = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    df["Date"] = timestamp.date()

    # === Update history file ===
    print("🗂️ Updating ESG risk history...")
    try:
        history_df = pd.read_csv(HISTORY_FILE)
        full_history = pd.concat([history_df, df], ignore_index=True)
    except FileNotFoundError:
        print("🎒 No previous history found. Creating new history.")
        full_history = df.copy()

    full_history.to_csv(HISTORY_FILE, index=False)

    # === Analyze ESG Risk Changes ===
    print("🔍 Checking for significant ESG risk changes...")
    try:
        full_history["Timestamp"] = pd.to_datetime(full_history["Timestamp"])
    except Exception as e:
        print("❌ Error parsing Timestamp column:", e)
        exit()

    if full_history["Timestamp"].nunique() < 2:
        print("ℹ️ Not enough entries yet to detect changes.")
    else:
        deltas = []
        for symbol in full_history["Symbol"].unique():
            symbol_scores = full_history[full_history["Symbol"] == symbol].sort_values("Timestamp")
            if len(symbol_scores) >= 2:
                latest = symbol_scores.iloc[-1]
                previous = symbol_scores.iloc[-2]
                delta = latest["Predicted ESG Risk Score"] - previous["Predicted ESG Risk Score"]
                deltas.append({
                    "Symbol": symbol,
                    "Previous Timestamp": previous["Timestamp"],
                    "Latest Timestamp": latest["Timestamp"],
                    "Previous Score": previous["Predicted ESG Risk Score"],
                    "Latest Score": latest["Predicted ESG Risk Score"],
                    "Delta": delta
                })

        delta_df = pd.DataFrame(deltas)

        if not delta_df.empty:
            delta_df.to_csv(DELTA_LOG_FILE, mode='a', index=False, header=not os.path.exists(DELTA_LOG_FILE))

            for _, row in delta_df.iterrows():
                full_history.loc[
                    (full_history["Symbol"] == row["Symbol"]) &
                    (full_history["Timestamp"] == row["Latest Timestamp"]),
                    "Delta"
                ] = row["Delta"]

            full_history.to_csv(HISTORY_FILE, index=False)

            risk_alerts = delta_df[delta_df["Delta"].abs() >= ALERT_THRESHOLD].sort_values("Delta", ascending=False).head(TOP_N)

            if risk_alerts.empty:
                print("✅ No significant ESG risk differences found.")
            else:
                print("📤 Sending alert email...")

                sender = "irabadyal8@gmail.com"
                receiver = "irabadyal8@gmail.com"
                password = "xxx!"  # Handle securely

                msg = MIMEMultipart("alternative")
                msg["Subject"] = "🚨 ESG Risk Alert: Significant Differences Detected"
                msg["From"] = sender
                msg["To"] = receiver

                text_body = f"Significant ESG Risk Score changes (Δ > {ALERT_THRESHOLD}):\n\n"
                for _, row in risk_alerts.iterrows():
                    text_body += f"{row['Symbol']}: {row['Previous Score']:.2f} → {row['Latest Score']:.2f} (Δ {row['Delta']:.2f})\n"

                msg.attach(MIMEText(text_body, "plain"))

                try:
                    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                        server.login(sender, password)
                        server.sendmail(sender, receiver, msg.as_string())
                    print("✅ Email sent successfully!")
                except Exception as e:
                    print(f"❌ Failed to send email: {e}")

# === Optional Gradio Interface ===
if len(sys.argv) > 1 and sys.argv[1] == "gradio_ui":
    features = ["Environment Risk Score", "Governance Risk Score", "Social Risk Score",
                "Controversy Score", "ESG Risk Percentile", "News Sentiment Score"]

    def predict_esg(company_name):
        row = df[df["Name"] == company_name]
        if row.empty:
            return f"❌ Company '{company_name}' not found in the dataset."

        input_row = row.iloc[0]
        input_features = []
        missing_features = []

        for f in features:
            val = input_row.get(f)
            if pd.isna(val):
                missing_features.append(f)
                input_features.append(0.0 if f == "News Sentiment Score" else df[f].mean())
            else:
                input_features.append(float(val))

        if len(missing_features) >= 2:
            return f"⚠️ Cannot predict: Missing values for multiple features: {', '.join(missing_features)}."

        try:
            input_array = np.array(input_features).reshape(1, -1)
            pred = model.predict(input_array)[0]
            return (
                f"✅ Predicted ESG Risk Score for **{company_name}**: {pred:.2f} "
                f"(missing: {', '.join(missing_features) if missing_features else 'None'})"
            )
        except Exception as e:
            return f"❌ Prediction error: {e}"

    company_list = df["Name"].unique().tolist()
    gr.Interface(
        fn=predict_esg,
        inputs=gr.Dropdown(choices=company_list, label="Select a Company"),
        outputs="text",
        title="📊 ESG Risk Predictor"
    ).launch(share=True)
