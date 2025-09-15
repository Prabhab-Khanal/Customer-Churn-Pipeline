

---

# Customer Churn Prediction Pipeline

This project builds a **Customer Churn Prediction Pipeline** using the [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
It demonstrates how to design a full machine learning workflow: **ingestion → preprocessing → training → prediction → storage → orchestration**.

---

## Project Structure

```
customer_churn_pipeline/
│── data/
│   ├── raw/               # original churn.csv dataset
│   └── processed/         # cleaned dataset (churn_clean.csv)
│── models/                # trained model + preprocessing pipeline
│── notebooks/             # Jupyter notebooks for EDA
│── src/                   # source code
│   ├── ingest.py          # ingest raw data
│   ├── transform.py       # preprocess data (encode, scale, clean)
│   ├── train.py           # train ML models
│   ├── predict.py         # generate predictions & save to DB
│   ├── query_predictions.py # check stored predictions
│   └── flow.py            # orchestrate full pipeline (Prefect)
│── db/                    # SQLite database (churn.db)
│── requirements.txt       # dependencies
│── README.md              # project documentation
│── .gitignore             # files ignored by Git
```

---

##  How It Works

1. **Ingest Data**

   * Load raw `churn.csv` into `data/raw/`.

2. **Transform Data**

   * Drop unused columns (`customerID`).
   * Convert `TotalCharges` → numeric.
   * Encode categorical features.
   * Scale numerical features.
   * Save clean dataset → `data/processed/churn_clean.csv`.
   * Save preprocessor → `models/preprocessor.pkl`.

3. **Train Model**

   * Train Logistic Regression, RandomForest, and XGBoost.
   * Compare metrics (Accuracy, F1, AUC).
   * Save best model → `models/churn_model.pkl`.

4. **Predict**

   * Load saved model + preprocessor.
   * Score new customer data (`data/raw/new_customers.csv`).
   * Save predictions to SQLite (`db/churn.db`, table: `churn_predictions`).

5. **Query Predictions**

   * Fetch stored predictions from database for analysis.

6. **Orchestrate**

   * Prefect flow (`flow.py`) automates the entire pipeline:

     ```
     ingest → transform → train → predict
     ```

---

##  Installation

```bash
git clone https://github.com/Prabhab-Khanal/Customer-Churn-Pipeline.git
cd customer_churn_pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate    # Mac/Linux
venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

---

##  Usage

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

---

##  Dataset

* **Name:** Telco Customer Churn Dataset
* **Records:** 7,043 customers
* **Features:** 21
* **Target:** `Churn (Yes/No)`

---

##  Features

* End-to-end modular ML pipeline
* Multiple models: Logistic Regression, RandomForest, XGBoost
* SQLite storage for predictions
* Prefect orchestration for automation
* Easy to extend to PostgreSQL or API deployment

---


