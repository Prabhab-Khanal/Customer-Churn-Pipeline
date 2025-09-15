from prefect import flow, task
from ingest import ingest_data
from transform import transform_and_save
from train import train_and_evaluate
from predict import predict_and_save

@task
def ingest_task():
    return ingest_data()

@task
def transform_task():
    transform_and_save()

@task
def train_task():
    train_and_evaluate()

@task
def predict_task():
    predict_and_save()

@flow
def churn_pipeline():
    print(" Starting Churn Pipeline")
    ingest_task()
    transform_task()
    train_task()
    predict_task()
    print(" Pipeline finished!")

if __name__ == "__main__":
    churn_pipeline()
