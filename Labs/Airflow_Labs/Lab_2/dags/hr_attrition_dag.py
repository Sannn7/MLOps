# File: dags/main.py
from __future__ import annotations

import os
import sys
import pendulum

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.utils.email import send_email


# ---- Fix imports: add project root (Lab_2/) to sys.path, not dags/ ----


DAGS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(DAGS_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.model_development import (
    load_data,
    data_preprocessing,
    separate_data_outputs,
    build_model,
    load_model,
)


def notify_failure(context):
    ti = context["task_instance"]
    dag_run = context.get("dag_run")
    exception = context.get("exception")

    dag_id = dag_run.dag_id if dag_run else ti.dag_id
    run_id = dag_run.run_id if dag_run else "unknown"

    subject = f"Airflow DAG Failed: {dag_id}"
    html_content = f"""
    <html>
      <body style="font-family: Arial; padding: 16px;">
        <h2 style="color:#c00;">Pipeline Failure</h2>
        <p><b>DAG:</b> {dag_id}</p>
        <p><b>Task:</b> {ti.task_id}</p>
        <p><b>Run ID:</b> {run_id}</p>
        <p><b>Error:</b> {str(exception)}</p>
        <p>Airflow UI: http://localhost:8080</p>
      </body>
    </html>
    """

    send_email(
        to=["YOUR_EMAIL@gmail.com"],
        subject=subject,
        html_content=html_content,
    )


default_args = {
    "start_date": pendulum.datetime(2024, 1, 1, tz="UTC"),
    "retries": 1,
    "retry_delay": pendulum.duration(minutes=5),
}


with DAG(
    dag_id="HR_Attrition_Pipeline",
    default_args=default_args,
    description="Employee Attrition Prediction Pipeline",
    schedule="@daily",
    catchup=False,
    tags=["hr", "ml", "attrition"],
    max_active_runs=1,
) as dag:

    initialize = BashOperator(
        task_id="initialize_pipeline",
        bash_command="echo 'Starting HR Attrition Pipeline...'",
    )

    load_hr_data = PythonOperator(
        task_id="load_hr_data",
        python_callable=load_data,
    )

    preprocess = PythonOperator(
        task_id="preprocess_and_engineer_features",
        python_callable=data_preprocessing,
        op_args=[load_hr_data.output],
    )

    split = PythonOperator(
        task_id="prepare_train_test_split",
        python_callable=separate_data_outputs,
        op_args=[preprocess.output],
    )

    train = PythonOperator(
        task_id="train_and_select_best_model",
        python_callable=build_model,
        op_args=[split.output, "hr_attrition_model.pkl"],
        on_failure_callback=notify_failure,
    )

    evaluate = PythonOperator(
        task_id="evaluate_model_performance",
        python_callable=load_model,
        op_args=[split.output, "hr_attrition_model.pkl"],
        on_failure_callback=notify_failure,
    )

    trigger_dashboard = TriggerDagRunOperator(
        task_id="trigger_flask_dashboard",
        trigger_dag_id="HR_Analytics_Dashboard",
        conf={"message": "Pipeline completed"},
        reset_dag_run=False,
        wait_for_completion=False,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    initialize >> load_hr_data >> preprocess >> split >> train >> evaluate >> trigger_dashboard