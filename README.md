# Realtime ESG Risk Score Model with News-Driven Sentiment Enhancement

## 📌 Project Summary
This project is a **Realtime ESG Risk Assessment Tool** that blends structured ESG datasets with real-time news sentiment to dynamically score public companies on ESG (Environmental, Social, Governance) risk. It uses a pretrained **XGBoost regression model** and OpenAI’s FinBERT model for sentiment analysis, all deployed in a user-friendly Streamlit application.

## 🌟 Key Features
- ✅ Real-time ESG sentiment fetched from recent news headlines via NewsAPI
- ✅ FinBERT-based sentiment classification on ESG-related news articles
- ✅ Predictive ESG risk score generated using a trained XGBoost model
- ✅ Comparison of predicted ESG scores vs original scores in dataset
- ✅ Search and predict ESG scores for any company name or ticker
- ✅ Visualization of ESG risk distribution and risk level breakdown
- ✅ Delta column to understand shift between model score and baseline score
- ✅ Streamlit web app interface with live prediction and analytics

## 🛠️ Tech Stack
- Python 3.10+
- Streamlit (for the UI)
- XGBoost (for regression model)
- HuggingFace Transformers (FinBERT for sentiment analysis)
- NewsAPI (real-time headline feed)
- Pandas, NumPy, Seaborn, Matplotlib (data processing + plotting)

## 📁 Project Structure
```
├── esg_model.py              # Streamlit application frontend
├── train_model.py           # XGBoost model training script
├── sp500_esg.csv            # Input ESG dataset (S&P 500 companies)
├── esg_model.pkl            # Pretrained XGBoost model file
├── .streamlit/secrets.toml  # API key configuration
```

## 📊 Dataset Overview
The included dataset (`sp500_esg.csv`) contains ESG data for S&P 500 companies, with columns such as:
- Company Name and Symbol
- Total ESG Risk Score
- Environment, Governance, and Social sub-scores
- ESG Risk Percentile
- Controversy Score
- Industry and Sector metadata

## 🔍 What the Code Does
### `esg_model.py`
- Loads ESG data and a pretrained XGBoost model
- Fetches recent ESG news (past ~3 days) using NewsAPI
- Uses FinBERT to classify news headline sentiment (positive, neutral, negative)
- Maps this sentiment to ESG impact score per company
- Predicts new ESG risk scores using the XGBoost model
- Compares predicted scores with original ESG risk scores
- Allows user to:
  - Search companies and compare risk levels
  - Manually enter a company name and estimate its risk sentiment
  - Visualize distribution and breakdown of risk levels
  - Download the full ESG results

### `train_model.py`
- Loads and cleans the ESG dataset
- Adds sentiment columns (if available)
- Splits data into training/testing sets
- Trains and saves an XGBoost regression model to predict ESG Risk Score

## 📈 Model Features Used
The XGBoost model uses the following features:
- Environment Risk Score
- Governance Risk Score
- Social Risk Score
- Controversy Score
- ESG Risk Percentile
- News Sentiment Score (derived from real-time news)

## ⚙️ How It Works (Simplified)
1. ESG scores + real-time sentiment = combined feature set
2. XGBoost predicts new ESG risk score
3. App compares this with original ESG score
4. Shows Risk Level (Negligible → Severe)
5. Allows interaction and filtering by company, score, or input

## 📬 News Coverage Timeframe
- ESG news headlines fetched from the **last 3 days** using NewsAPI
- Sentiment classification runs on headlines only (no full articles)

## 🚀 Getting Started
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Add your NewsAPI key in:
```
.streamlit/secrets.toml
```
```toml
NEWS_API_KEY = "your_key_here"
```

3. Launch the Streamlit app:
```bash
streamlit run esg_model.py
```

## 📤 Output
The app exports a CSV file with:
- Company Name, Symbol
- Predicted ESG Risk Score
- Total ESG Risk Score
- Risk Level label
- Risk Score Delta

