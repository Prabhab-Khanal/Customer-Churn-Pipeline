import sqlite3
import pandas as pd

DB_PATH = "db/churn.db"

def fetch_predictions():
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM churn_predictions ORDER BY prediction_date DESC"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

if __name__ == "__main__":
    predictions = fetch_predictions()
    print("📊 Stored Predictions in DB:")
    print(predictions.head())
