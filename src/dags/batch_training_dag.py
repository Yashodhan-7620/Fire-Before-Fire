"""
Airflow DAG: daily batch pipeline.
data intake (already continuous via the ingestion API) -> preprocess
-> train -> (model registered + threshold alerts inside train.py).

Copy/symlink this file into your Airflow DAGs folder
(usually ~/airflow/dags/) to have Airflow pick it up.
Runs once a day at 23:30, i.e. "at the end of the day" per the brief.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator


def _preprocess():
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from src.pipelines.preprocess import run
    run()


def _train():
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from src.pipelines.train import main
    main()


default_args = {
    "owner": "fire-mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="fire_risk_batch_training",
    description="Daily: preprocess sensor data, retrain forecaster/anomaly model, check thresholds",
    default_args=default_args,
    schedule_interval="30 23 * * *",   # 23:30 every day
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["mlops", "fire-risk"],
) as dag:

    preprocess_task = PythonOperator(
        task_id="preprocess_features",
        python_callable=_preprocess,
    )

    train_task = PythonOperator(
        task_id="train_and_register_model",
        python_callable=_train,
    )

    preprocess_task >> train_task
