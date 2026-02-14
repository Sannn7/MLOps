# File: src/model_development.py
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------- Paths (single, consistent source of truth) ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # <project>/src/model_development.py -> <project>
WORKING_DIR = PROJECT_ROOT / "working_data"
MODEL_DIR = PROJECT_ROOT / "model"
DATA_DIR = PROJECT_ROOT / "data"
HR_PATH = DATA_DIR / "HR.csv"

WORKING_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"WORKING_DIR: {WORKING_DIR}")
print(f"MODEL_DIR: {MODEL_DIR}")
print(f"DATA_DIR: {DATA_DIR}")
print(f"HR_PATH: {HR_PATH}")


def load_hr_data() -> str:
    """
    Loads HR.csv from disk, saves it as a pickle (raw.pkl) to WORKING_DIR,
    and returns the pickle path (string) so XCom stays small and safe.
    """
    if not HR_PATH.exists():
        raise FileNotFoundError(f"HR.csv not found at: {HR_PATH}")

    df = pd.read_csv(HR_PATH)

    out_path = WORKING_DIR / "raw.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(df, f)

    print(f"✓ Loaded HR.csv and saved raw pickle to: {out_path}")
    return str(out_path)


def data_preprocessing(file_path: str) -> str:
    """
    HR-specific preprocessing:
    - Feature engineering
    - Encode categorical variables
    - Scale numeric features
    - Stratified split
    Saves preprocessed.pkl and transformer.pkl, returns preprocessed.pkl path.
    """
    in_path = Path(file_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input pickle not found: {in_path}")

    with open(in_path, "rb") as f:
        df = pickle.load(f)

    # === FEATURE ENGINEERING ===
    df["satisfaction_category"] = pd.cut(
        df["satisfaction_level"],
        bins=[0, 0.3, 0.6, 1.0],
        labels=["low", "medium", "high"],
        include_lowest=True,
    )

    df["overworked"] = (df["average_montly_hours"] > 250).astype(int)
    df["projects_per_year"] = df["number_project"] / (df["time_spend_company"] + 1)

    df["high_performer_risk"] = (
        (df["last_evaluation"] > 0.7) & (df["satisfaction_level"] < 0.5)
    ).astype(int)

    # === PREPARE FEATURES ===
    numeric_features = [
        "satisfaction_level",
        "last_evaluation",
        "number_project",
        "average_montly_hours",
        "time_spend_company",
        "projects_per_year",
    ]

    binary_features = [
        "Work_accident",
        "promotion_last_5years",
        "overworked",
        "high_performer_risk",
    ]

    categorical_features = [
        "sales",
        "salary",
        "satisfaction_category",
    ]

    X = df[numeric_features + binary_features + categorical_features].copy()
    y = df["left"]

    # === TRAIN-TEST SPLIT (STRATIFIED) ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # === PREPROCESSOR ===
    ct = make_column_transformer(
        (StandardScaler(), numeric_features),
        (OneHotEncoder(drop="first", sparse_output=False), categorical_features),
        remainder="passthrough",
    )

    X_train_tr = ct.fit_transform(X_train)
    X_test_tr = ct.transform(X_test)

    # === SAVE OUTPUTS ===
    out_path = WORKING_DIR / "preprocessed.pkl"
    with open(out_path, "wb") as f:
        pickle.dump((X_train_tr, X_test_tr, y_train.values, y_test.values), f)

    transformer_path = MODEL_DIR / "transformer.pkl"
    with open(transformer_path, "wb") as f:
        pickle.dump(ct, f)

    print(f"✓ Preprocessing complete. Saved to {out_path}")
    return str(out_path)


def separate_data_outputs(file_path: str) -> str:
    """
    Passthrough; kept so the DAG composes cleanly.
    """
    return file_path


def build_model(file_path: str, filename: str) -> str:
    """
    Train multiple models and save the best one.
    """
    in_path = Path(file_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Preprocessed pickle not found: {in_path}")

    with open(in_path, "rb") as f:
        X_train, X_test, y_train, y_test = pickle.load(f)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=42
        ),
    }

    best_model = None
    best_score = -1.0
    best_name = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        test_score = model.score(X_test, y_test)
        if test_score > best_score:
            best_score = test_score
            best_model = model
            best_name = name

    model_path = MODEL_DIR / filename
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    metadata = {
        "model_name": best_name,
        "test_accuracy": float(best_score),
        "training_date": pd.Timestamp.now().isoformat(),
    }

    metadata_path = MODEL_DIR / "model_metadata.pkl"
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    print(f"✓ Best model: {best_name} (accuracy={best_score:.4f})")
    print(f"✓ Model saved to {model_path}")
    return str(model_path)


def load_model(file_path: str, filename: str) -> dict:
    """
    Load model, evaluate performance, and save metrics.
    Returns metrics dict (small) for downstream use.
    """
    in_path = Path(file_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Preprocessed pickle not found: {in_path}")

    with open(in_path, "rb") as f:
        X_train, X_test, y_train, y_test = pickle.load(f)

    model_path = MODEL_DIR / filename
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    else:
        roc_auc = None

    test_accuracy = model.score(X_test, y_test)

    cm = confusion_matrix(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "test_accuracy": float(test_accuracy),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "evaluation_date": pd.Timestamp.now().isoformat(),
    }

    metrics_path = MODEL_DIR / "evaluation_metrics.pkl"
    with open(metrics_path, "wb") as f:
        pickle.dump(metrics, f)

    print(f"✓ Metrics saved to {metrics_path}")
    return metrics