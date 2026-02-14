# File: dags/flask_api.py
from __future__ import annotations
import os
import base64
import pendulum
import requests
from airflow import DAG
from airflow.operators.python import PythonOperator
from flask import Flask, redirect, render_template

# ---------- Config ----------
WEBSERVER = os.getenv("AIRFLOW_WEBSERVER", "http://localhost:8080")
AF_USER = os.getenv("AIRFLOW_USERNAME", "airflow")
AF_PASS = os.getenv("AIRFLOW_PASSWORD", "airflow")
TARGET_DAG_ID = "HR_Attrition_Pipeline"  # Must match main DAG name

# Set up paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to find templates folder
TEMPLATE_DIR = os.path.join(os.path.dirname(BASE_DIR), "templates")

print(f"Looking for templates in: {TEMPLATE_DIR}")

# ---------- Flask app ----------
app = Flask(__name__, template_folder=TEMPLATE_DIR)

def get_latest_run_info():
    """Query Airflow API for latest DAG run status"""
    url = f"{WEBSERVER}/api/v1/dags/{TARGET_DAG_ID}/dagRuns"
    
    info = {
        "state": "unknown",
        "run_id": "N/A",
        "logical_date": "N/A",
        "start_date": "N/A",
        "end_date": "N/A",
        "note": ""
    }
    
    # Basic auth for Airflow API
    auth_bytes = f"{AF_USER}:{AF_PASS}".encode("ascii")
    base64_auth = base64.b64encode(auth_bytes).decode("ascii")
    headers = {
        "Accept": "application/json",
        "Authorization": f"Basic {base64_auth}"
    }
    
    try:
        r = requests.get(url, headers=headers, timeout=10)
        
        if r.status_code != 200:
            info["note"] = f"API Error {r.status_code}"
            return False, info
        
        data = r.json()
        runs = data.get("dag_runs", [])
        
        if not runs:
            info["note"] = "No DAG runs found. Trigger the pipeline first!"
            return False, info
        
        # Get most recent run
        run = runs[0]
        state = run.get("state", "unknown")
        
        info.update({
            "state": state,
            "run_id": run.get("dag_run_id", "N/A"),
            "logical_date": run.get("logical_date", "N/A"),
            "start_date": run.get("start_date", "N/A"),
            "end_date": run.get("end_date", "N/A"),
        })
        
        # Handle different states
        if state in ["queued", "running"]:
            info["note"] = "Pipeline is currently processing HR data..."
            return False, info
        
        return state == "success", info
        
    except Exception as e:
        info["note"] = f"Connection error: {str(e)}"
        return False, info

@app.route("/")
def index():
    """Redirect based on pipeline status"""
    ok, _ = get_latest_run_info()
    return redirect("/success" if ok else "/failure")

@app.route("/success")
def success():
    """Show success page"""
    _, info = get_latest_run_info()
    return render_template("success.html", **info)

@app.route("/failure")
def failure():
    """Show failure/loading page"""
    _, info = get_latest_run_info()
    return render_template("failure.html", **info)

@app.route("/health")
def health():
    """Health check endpoint"""
    return "OK", 200

def start_flask_app():
    """Start Flask server"""
    print(f"Starting Flask dashboard on http://0.0.0.0:5555", flush=True)
    print(f"Monitoring DAG: {TARGET_DAG_ID}", flush=True)
    print(f"Template directory: {TEMPLATE_DIR}", flush=True)
    app.run(host="0.0.0.0", port=5555, debug=False, use_reloader=False)

# ---------- DAG Definition ----------
default_args = {
    "start_date": pendulum.datetime(2024, 1, 1, tz="UTC"),
    "retries": 0,
}

dag = DAG(
    dag_id="HR_Analytics_Dashboard",
    default_args=default_args,
    schedule=None,  # Triggered by main DAG
    catchup=False,
    is_paused_upon_creation=False,
    tags=["dashboard", "monitoring"],
)

start_flask_task = PythonOperator(
    task_id="start_flask_dashboard",
    python_callable=start_flask_app,
    dag=dag,
)