# src/predict.py
import pandas as pd
import joblib
import sqlite3
import os
from datetime import datetime

# Paths
MODEL_PATH = "models/churn_model.pkl"
PIPELINE_PATH = "models/preprocessor.pkl"
INPUT_PATH = "data/raw/new_customers.csv"   # new customers to score
DB_PATH = "db/churn.db"

def load_artifacts():
    """Load trained model and preprocessor."""
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PIPELINE_PATH)
    return model, preprocessor

def load_new_data():
    """Load new customer data to score."""
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"{INPUT_PATH} not found. Please add a CSV with new customers.")
    df = pd.read_csv(INPUT_PATH)

    # Drop customerID if present, but keep for output
    customer_ids = df["customerID"] if "customerID" in df.columns else None
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    return df, customer_ids

def predict_and_save():
    """Generate predictions and save to SQLite DB."""
    model, preprocessor = load_artifacts()
    df, customer_ids = load_new_data()

    # Transform features
    X_processed = preprocessor.transform(df)

    # Predict churn & probability
    probs = model.predict_proba(X_processed)[:, 1]  # probability of churn
    preds = (probs >= 0.5).astype(int)

    # Build results dataframe
    results = pd.DataFrame({
        "customerID": customer_ids if customer_ids is not None else range(len(preds)),
        "predicted_label": preds,
        "probability": probs,
        "prediction_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

    # Save results to SQLite
    os.makedirs("db", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    results.to_sql("churn_predictions", conn, if_exists="append", index=False)
    conn.close()

    print(f"✅ Predictions saved to {DB_PATH} (table: churn_predictions)")
    print(results.head())

if __name__ == "__main__":
    predict_and_save()
