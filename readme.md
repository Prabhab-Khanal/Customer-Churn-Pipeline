     ingest → transform → train → predict
     ```

---

## Installation

## Installation

```bash
git clone <your-repo-url>
cd customer_churn_pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate    # Mac/Linux
venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

## ▶ Usage

### Run full pipeline

```bash
python src/flow.py
```

### Run steps individually

```bash
python src/ingest.py
python src/transform.py
python src/train.py
python src/predict.py
python src/query_predictions.py
```

## Dataset

- **Name:** Telco Customer Churn Dataset
- **Records:** 7,043 customers
- **Features:** 21
- **Target:** `Churn (Yes/No)`

---

## Features

- End-to-end modular ML pipeline
- Multiple models: Logistic Regression, RandomForest, XGBoost
- SQLite storage for predictions
- Prefect orchestration for automation
- Easy to extend to PostgreSQL or API deployment

---

```

```
