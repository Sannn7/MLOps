# File: dags/main.py
from __future__ import annotations
import pendulum
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.email import EmailOperator
from airflow.utils.trigger_rule import TriggerRule  

# Import your custom functions
import sys
import os
# Add the dags directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

from src.model_development import (
    load_data,
    data_preprocessing,
    separate_data_outputs,
    build_model,
    load_model,
)

# ---------- Email Notification Functions ----------
def notify_failure(context):
    """Send email on task failure"""
    task_instance = context.get('task_instance')
    dag_run = context.get('dag_run')
    exception = context.get('exception')
    
    html_content = f"""
    <html>
        <body style="font-family: Arial; padding: 20px;">
            <div style="background-color: #fee; border-left: 4px solid #f00; padding: 20px;">
                <h2 style="color: #c00;">⚠️ HR Pipeline Failure Alert</h2>
                <p><strong>DAG:</strong> {dag_run.dag_id}</p>
                <p><strong>Task:</strong> {task_instance.task_id}</p>
                <p><strong>Run ID:</strong> {dag_run.run_id}</p>
                <p><strong>Error:</strong> {str(exception)}</p>
                <hr>
                <p>Check Airflow UI for details: http://localhost:8080</p>
            </div>
        </body>
    </html>
    """
    
    email_task = EmailOperator(
        task_id='failure_email',
        to='YOUR_EMAIL@gmail.com',  # CHANGE THIS
        subject=f'🚨 {dag_run.dag_id} Failed',
        html_content=html_content,
        dag=context['dag']
    )
    return email_task.execute(context)

# ---------- Default Args ----------
default_args = {
    "start_date": pendulum.datetime(2024, 1, 1, tz="UTC"),
    "retries": 1,
    "retry_delay": pendulum.duration(minutes=5),
}

# ---------- DAG Definition ----------
dag = DAG(
    dag_id="HR_Attrition_Pipeline",
    default_args=default_args,
    description="Employee Attrition Prediction Pipeline",
    schedule="@daily",  # Change to None for manual trigger only
    catchup=False,
    tags=["hr", "ml", "attrition"],
    owner_links={"Your Name": "https://github.com/YOUR_USERNAME"},
    max_active_runs=1,
)

# ---------- Tasks ----------
owner_task = BashOperator(
    task_id="initialize_pipeline",
    bash_command="echo 'Starting HR Attrition Pipeline...'",
    owner="Your Name",
    dag=dag,
)

load_data_task = PythonOperator(
    task_id="load_hr_data",
    python_callable=load_data,
    dag=dag,
)

data_preprocessing_task = PythonOperator(
    task_id="preprocess_and_engineer_features",
    python_callable=data_preprocessing,
    op_args=[load_data_task.output],
    dag=dag,
)

separate_data_outputs_task = PythonOperator(
    task_id="prepare_train_test_split",
    python_callable=separate_data_outputs,
    op_args=[data_preprocessing_task.output],
    dag=dag,
)

build_save_model_task = PythonOperator(
    task_id="train_and_select_best_model",
    python_callable=build_model,
    op_args=[separate_data_outputs_task.output, "hr_attrition_model.pkl"],
    on_failure_callback=notify_failure,  # Email on failure
    dag=dag,
)

load_model_task = PythonOperator(
    task_id="evaluate_model_performance",
    python_callable=load_model,
    op_args=[separate_data_outputs_task.output, "hr_attrition_model.pkl"],
    on_failure_callback=notify_failure,  # Email on failure
    dag=dag,
)

trigger_dag_task = TriggerDagRunOperator(
    task_id="trigger_flask_dashboard",
    trigger_dag_id="HR_Analytics_Dashboard",
    conf={"message": "Pipeline completed"},
    reset_dag_run=False,
    wait_for_completion=False,
    trigger_rule=TriggerRule.ALL_DONE,
    dag=dag,
)

# ---------- Task Dependencies ----------
owner_task >> load_data_task >> data_preprocessing_task >> \
    separate_data_outputs_task >> build_save_model_task >> \
    load_model_task >> trigger_dag_task