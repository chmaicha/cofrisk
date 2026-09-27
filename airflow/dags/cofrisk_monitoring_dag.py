from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator


def validate_data():
    print("Validation des données production...")


def detect_drift():
    print("Détection du data drift...")


def evaluate_drift():
    print("Évaluation du drift...")


default_args = {
    "owner": "cofrisk",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="cofrisk_monitoring",
    default_args=default_args,
    description="CofRisk production monitoring pipeline",
    schedule="0 2 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["cofrisk", "monitoring", "ml"],
) as dag:

    validate = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
    )

    drift = PythonOperator(
        task_id="detect_drift",
        python_callable=detect_drift,
    )

    evaluate = PythonOperator(
        task_id="evaluate_drift",
        python_callable=evaluate_drift,
    )

    validate >> drift >> evaluate