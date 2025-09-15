# src/transform.py
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

RAW_PATH = "data/raw/churn.csv"
PROCESSED_PATH = "data/processed/churn_clean.csv"
PIPELINE_PATH = "models/preprocessor.pkl"

def load_data():
    """Load raw churn dataset."""
    return pd.read_csv(RAW_PATH)

def preprocess(df):
    """Clean and preprocess churn dataset."""

    # 1. Drop customerID
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # 2. Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # 3. Target variable
    y = df["Churn"].map({"Yes": 1, "No": 0})
    X = df.drop(columns=["Churn"])

    # 4. Identify categorical & numerical features
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # 5. Build preprocessing pipeline
    categorical = OneHotEncoder(handle_unknown="ignore")
    numeric = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical, cat_cols),
            ("numeric", numeric, num_cols)
        ]
    )

    # 6. Transform features
    X_processed = preprocessor.fit_transform(X)

    return X_processed, y, preprocessor

def transform_and_save():
    """Main function to preprocess data and save outputs."""
    df = load_data()
    X_processed, y, preprocessor = preprocess(df)

    # Save processed dataset
    X_df = pd.DataFrame(
        X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed
    )
    X_df["Churn"] = y.values

    os.makedirs("data/processed", exist_ok=True)
    X_df.to_csv(PROCESSED_PATH, index=False)

    # Save preprocessing pipeline
    os.makedirs("models", exist_ok=True)
    joblib.dump(preprocessor, PIPELINE_PATH)

    print(f" Data transformed and saved to {PROCESSED_PATH}")
    print(f" Preprocessor saved to {PIPELINE_PATH}")

if __name__ == "__main__":
    transform_and_save()
