import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta

from src.lab import load_data, feature_engineering, data_preprocessing, build_save_model, load_model_elbow, visualize_clusters

# Define default arguments for your DAG
default_args = {
    'owner': 'SANIKA KILLEKAR',
    'start_date': datetime(2026, 1, 31),
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

# Create a DAG instance
with DAG(
    dag_id="Airflow_Lab1",
    default_args=default_args,
    description="Spotify clustering DAG (feature engg + preprocessing + kmeans + elbow + visualization)",
    schedule=None,
    catchup=False,
) as dag:

    # Task to load data
    load_data_task = PythonOperator(
        task_id='load_data_task',
        python_callable=load_data,
    )

    # Feature engineering task - uses XCom to get data from load_data_task
    feature_engineering_task = PythonOperator(
        task_id="feature_engineering_task",
        python_callable=lambda **context: feature_engineering(
            context['ti'].xcom_pull(task_ids='load_data_task')
        ),
    )

    # Data preprocessing task - uses XCom to get data from feature_engineering_task
    data_preprocessing_task = PythonOperator(
        task_id="data_preprocessing_task",
        python_callable=lambda **context: data_preprocessing(
            context['ti'].xcom_pull(task_ids='feature_engineering_task')
        ),
    )

    # Build and save model task - uses XCom to get data from data_preprocessing_task
    build_save_model_task = PythonOperator(
        task_id='build_save_model_task',
        python_callable=lambda **context: build_save_model(
            context['ti'].xcom_pull(task_ids='data_preprocessing_task'),
            "model.sav"
        ),
    )

    # Load model task - uses XCom to get SSE list from build_save_model_task
    load_model_task = PythonOperator(
        task_id='load_model_task',
        python_callable=lambda **context: load_model_elbow(
            "model.sav",
            context['ti'].xcom_pull(task_ids='build_save_model_task')
        ),
    )

    # Visualization task - creates plots showing clustering results
    visualize_clusters_task = PythonOperator(
        task_id='visualize_clusters_task',
        python_callable=lambda **context: visualize_clusters(
            "model.sav",
            context['ti'].xcom_pull(task_ids='build_save_model_task')
        ),
    )

    # Set task dependencies
    load_data_task >> feature_engineering_task >> data_preprocessing_task >> build_save_model_task >> load_model_task >> visualize_clusters_task

# If this script is run directly, allow command-line interaction with the DAG
if __name__ == "__main__":
    dag.test()