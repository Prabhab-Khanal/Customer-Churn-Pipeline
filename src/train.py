# src/train.py
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

PROCESSED_PATH = "data/processed/churn_clean.csv"
MODEL_PATH = "models/churn_model.pkl"

def load_data():
    """Load processed dataset for training."""
    return pd.read_csv(PROCESSED_PATH)

def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics."""
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, preds)
    return acc, f1, auc

def train_and_evaluate():
    df = load_data()
    y = df["Churn"]
    X = df.drop(columns=["Churn"])

    # Split data (stratify keeps churn ratio same in train/test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Models to compare
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(
            use_label_encoder=False, eval_metric="logloss", random_state=42
        )
    }

    best_model, best_auc = None, 0

    # Train & evaluate each model
    for name, model in models.items():
        print(f"\n🔹 Training {name}...")
        model.fit(X_train, y_train)
        acc, f1, auc = evaluate_model(model, X_test, y_test)

        print(f"   Accuracy: {acc:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")

        if auc > best_auc:
            best_auc = auc
            best_model = model

    # Save best model
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"\n Best model saved: {best_model.__class__.__name__} → {MODEL_PATH}")

if __name__ == "__main__":
    train_and_evaluate()
