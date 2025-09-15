import pandas as pd

RAW_PATH = "data/raw/churn.csv"

def ingest_data():
    """Load raw churn dataset."""
    df = pd.read_csv(RAW_PATH)
    print(f" Ingested data with shape: {df.shape}")
    return df

if __name__ == "__main__":
    ingest_data()
