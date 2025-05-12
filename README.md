# peacepie
# 📘 ESG Risk Monitoring & Forecasting Platform

A real-time ESG intelligence tool designed for sustainable finance, enabling analysts to predict and monitor ESG risk through sentiment-aware modeling — aligned with SDG investment strategies and regulatory transparency goals.


---

## 🧠 Overview

This platform enables engineers and analysts to:

* Train an XGBoost model on ESG + news sentiment features
* Predict and track ESG risk scores over time
* Compare recent deltas in ESG risk across companies
* Expose an interactive user interface via Gradio
* Receive email alerts when risk changes cross defined thresholds

Built for extensibility and runtime flexibility, the pipeline allows toggling between batch training, inference, delta tracking, and UI deployment.

---

## 📁 Repository Structure

```
.
├── train_model.py            # Merges datasets, enriches with sentiment, trains XGBoost model
├── esg_model.py              # Main batch prediction + Gradio UI interface
├── esg_trend_agent.py        # Historical ESG delta checker with alert system
├── merged_esg_data.csv       # Combined dataset for prediction/training
├── esg_model.pkl             # Trained XGBoost regression model
├── esg_score_history.csv     # ESG scores over time
├── delta_log.csv             # Logged ESG risk changes
```

---

## 🔧 Requirements

```bash
pip install pandas numpy scikit-learn xgboost transformers newsapi-python fuzzywuzzy gradio
```

For improved `fuzzywuzzy` performance:

```bash
pip install python-Levenshtein
```

---

## 🛠️ Configuration

Set the following environment variables before running any script:

```python
os.environ["MODEL_PATH"] = "/path/to/esg_model.pkl"
os.environ["DATA_PATH"] = "/path/to/merged_esg_data.csv"
os.environ["HISTORY_FILE"] = "/path/to/esg_score_history.csv"
os.environ["DELTA_LOG_FILE"] = "/path/to/delta_log.csv"
os.environ["NEWS_API_KEY"] = "<your_newsapi_key>"
```

---

## 🚀 Scripts

### 1. `train_model.py`

* Merges raw ESG datasets (SP500 + custom)
* Enriches them with FinBERT sentiment from NewsAPI
* Trains XGBoost regression model
* Outputs: `esg_model.pkl`, `merged_esg_data.csv`

```bash
python3 train_model.py
```

---

### 2. `esg_model.py`

#### Runtime 1: Batch Inference

* Fetches real-time sentiment (if `sys.argv != 'gradio_ui'`)
* Predicts ESG risk scores
* Appends to `esg_score_history.csv`
* Triggers alert if risk delta exceeds threshold

```bash
python3 esg_model.py
```

#### Runtime 2: Gradio UI

Launches an interactive ESG risk predictor.

```bash
python3 esg_model.py gradio_ui
```

**Gradio App Features:**

* Dropdown company selector
* Graceful fallback if 1 feature is missing (imputes default/mean)
* Blocks prediction if 2+ features missing

---

### 3. `esg_trend_agent.py`

Agentic pipeline to:

* Recompute FinBERT sentiment in real time
* Compare historical vs. current predictions
* Append to history + log deltas
* Send alert email if deltas > threshold

```bash
python3 esg_trend_agent.py
```

---

## 📈 Model Inputs (Features)

* `Environment Risk Score`
* `Governance Risk Score`
* `Social Risk Score`
* `Controversy Score`
* `ESG Risk Percentile`
* `News Sentiment Score` *(from FinBERT)*

---

## 🛡️ Alerting Logic

* ESG risk delta > 10 triggers email alert
* Email sent via SMTP (Gmail SSL)
* Top 3 companies with highest delta included

---

## 🧪 Example Use Cases

* ESG trend tracking for S\&P 500
* Identifying emerging ESG issues from news
* Building ESG investment monitoring dashboards
* Realtime interfaces for risk analysts or compliance teams

---

## 🧠 Future Enhancements

* Add vector DB memory for persistent sentiment history
* Integrate GDELT for global news ingestion
* Allow user-input ESG values via UI + real-time prediction
* Dashboard integrations (e.g., Streamlit sharing / Grafana)

---

## 👨‍💻 Author & License

Created by \[Your Name] — MIT License. Feel free to fork, extend, and deploy.

---

## 🔗 Live Demo (Gradio)

> Launch via:

```bash
python3 esg_model.py gradio_ui
```

> Shareable link will be printed in console after launch.

![image](https://github.com/user-attachments/assets/8fb17e3c-040b-4977-8bd5-ece34f003937)
![image](https://github.com/user-attachments/assets/f9da8963-56b5-462c-93aa-8a5543f0288d)




---
