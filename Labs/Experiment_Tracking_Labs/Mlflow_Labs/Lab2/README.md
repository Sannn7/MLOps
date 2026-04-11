# MLflow Lab 2 — Titanic Survival Prediction

This lab demonstrates the MLflow experiment tracking workflow using the **Titanic Survival dataset**. It covers the full MLflow lifecycle: experiment tracking, multi-model comparison, model registration, stage transitions, and model serving.

---

## Dataset

**Titanic Survival Dataset** — loaded directly via `seaborn.load_dataset('titanic')`.

- **Task**: Binary classification (Survived: 0/1)
- **Features**: Passenger class, sex, age, siblings/spouses, parents/children, fare, embarkation port
- **Engineered Features**: `family_size`, `is_alone`
- **Split**: 60% train / 20% validation / 20% test (stratified)

---

## Lab Structure

```
Lab2/
├── data/                          # Dataset folder
├── mlruns/                        # MLflow tracking directory (auto-generated)
├── Result_screenshots/            # Screenshots of MLflow UI results
│   ├── mlflow_all_runs_experiment.png
│   ├── mlflow_metrics_comparison_chart.png
│   ├── mlflow_gradient_boosting_run_detail.png
│   ├── mlflow_artifact_confusion_matrix.png
│   ├── mlflow_artifact_eda_plots.png
│   ├── mlflow_model_registry.png
│   └── mlflow_version2_production_stage.png
├── starter.ipynb                  # Main implementation notebook
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

---

## MLflow Concepts Covered

| MLflow Concept | Implementation |
|---|---|
| `mlflow.set_experiment` | Named experiment: `titanic_survival_prediction` |
| `mlflow.start_run` | One run per model — 3 models, 6 total runs |
| `mlflow.log_param` | Model type, hyperparameters per model |
| `mlflow.log_metric` | AUC, Accuracy, F1 on both validation and test sets |
| `mlflow.log_artifact` | Confusion matrix, EDA plots, feature importance plots |
| `mlflow.pyfunc.log_model` | Model logged with input/output signature |
| `mlflow.search_runs` | Programmatic run comparison to select best model |
| `mlflow.register_model` | Best model registered as `titanic_survival_model` |
| `client.transition_model_version_stage` | Model promoted to **Production** stage |
| `mlflow.pyfunc.load_model` | Production model loaded for batch inference |
| `mlflow models serve` | REST API serving on port 5002 |

---

## Models Trained

Three models were trained and logged as separate MLflow runs:

| Model | Key Parameters |
|---|---|
| Logistic Regression | C=1.0, max_iter=500 |
| Random Forest | n_estimators=100, max_depth=6 |
| Gradient Boosting | n_estimators=100, learning_rate=0.1, max_depth=4 |

Each run logs: `val_auc`, `val_acc`, `val_f1`, `test_auc`, `test_acc`, `test_f1`

The best model selected by `test_auc` is automatically registered in the MLflow Model Registry and promoted to Production.

---

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Execute the Notebook

```bash
jupyter nbconvert --to notebook --execute starter.ipynb --output starter.ipynb
```

### 3. Launch MLflow UI

```bash
mlflow ui --port 5001
```

Open **http://127.0.0.1:5001** in your browser.

### 4. Serve the Production Model

```bash
mlflow models serve -m models:/titanic_survival_model/Production -h 0.0.0.0 -p 5002 --no-conda
```

### 5. Real-Time Inference

```python
import requests, json

url = 'http://localhost:5002/invocations'
payload = {"dataframe_split": X_test.to_dict(orient='split')}
response = requests.post(url, json=payload)
print(response.json())
```

---

## Screenshots

All result screenshots are stored in `Result_screenshots/`:

| Screenshot | Description |
|---|---|
| `mlflow_all_runs_experiment.png` | All 6 runs in the experiment |
| `mlflow_metrics_comparison_chart.png` | Metric comparison across all models |
| `mlflow_gradient_boosting_run_detail.png` | Params and metrics for gradient boosting run |
| `mlflow_artifact_confusion_matrix.png` | Confusion matrix artifact logged in MLflow |
| `mlflow_artifact_eda_plots.png` | EDA plots artifact logged in MLflow |
| `mlflow_model_registry.png` | Model registry showing titanic_survival_model |
| `mlflow_version2_production_stage.png` | Version 2 promoted to Production stage |

---

## Workflow

The `.github/workflows/` directory contains the CI/CD pipeline configuration for this repository. On every push to `main`, the workflow runs automated checks to validate the lab code and ensure reproducibility of the MLflow experiment.

---

## Dependencies

```
mlflow
scikit-learn
pandas
numpy
seaborn
matplotlib
cloudpickle
requests
jupyter
nbconvert
```
