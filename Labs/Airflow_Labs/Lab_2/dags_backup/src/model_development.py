# File: src/model_development.py
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORKING_DIR = os.path.join(PROJECT_ROOT, "working_data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")

# Create directories if they don't exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"WORKING_DIR: {WORKING_DIR}")
print(f"MODEL_DIR: {MODEL_DIR}")

def load_data() -> str:
    """Load HR CSV and persist raw dataframe to pickle"""
    csv_path = os.path.join(PROJECT_ROOT, "data", "HR.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"HR.csv not found at: {csv_path}")


def data_preprocessing(file_path: str) -> str:
    """
    HR-specific preprocessing:
    - Feature engineering (satisfaction categories, workload ratio)
    - Handle categorical variables (sales dept, salary)
    - Scale numeric features
    - Stratified split for imbalanced target
    """
    with open(file_path, "rb") as f:
        df = pickle.load(f)
    
    # === FEATURE ENGINEERING (YOUR CUSTOM ADDITIONS) ===
    
    # 1. Create satisfaction categories
    df['satisfaction_category'] = pd.cut(
        df['satisfaction_level'], 
        bins=[0, 0.3, 0.6, 1.0],
        labels=['low', 'medium', 'high']
    )
    
    # 2. Create workload indicator (overworked employees)
    df['overworked'] = (df['average_montly_hours'] > 250).astype(int)
    
    # 3. Projects per year ratio
    df['projects_per_year'] = df['number_project'] / (df['time_spend_company'] + 1)
    
    # 4. Identify high performers at risk (high eval, low satisfaction)
    df['high_performer_risk'] = (
        (df['last_evaluation'] > 0.7) & 
        (df['satisfaction_level'] < 0.5)
    ).astype(int)
    
    print("\n=== Feature Engineering Complete ===")
    print(f"Overworked employees: {df['overworked'].sum()}")
    print(f"High performers at risk: {df['high_performer_risk'].sum()}")
    
    # === PREPARE FEATURES ===
    
    # Define feature groups
    numeric_features = [
        'satisfaction_level', 
        'last_evaluation', 
        'number_project',
        'average_montly_hours', 
        'time_spend_company',
        'projects_per_year'
    ]
    
    binary_features = [
        'Work_accident', 
        'promotion_last_5years',
        'overworked',
        'high_performer_risk'
    ]
    
    categorical_features = [
        'sales',  # department
        'salary',
        'satisfaction_category'
    ]
    
    # Combine all features
    X = df[numeric_features + binary_features + categorical_features].copy()
    y = df['left']
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target balance: {y.value_counts(normalize=True).to_dict()}")
    
    # === TRAIN-TEST SPLIT (STRATIFIED) ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.25,  # 75-25 split
        random_state=42,
        stratify=y  # Keep same class distribution
    )
    
    # === BUILD PREPROCESSING PIPELINE ===
    
    # Create column transformer
    ct = make_column_transformer(
        (StandardScaler(), numeric_features),  # Scale numeric
        (OneHotEncoder(drop='first', sparse_output=False), categorical_features),  # Encode categorical
        remainder='passthrough'  # Keep binary features as-is
    )
    
    # Fit and transform
    X_train_tr = ct.fit_transform(X_train)
    X_test_tr = ct.transform(X_test)
    
    print(f"Transformed training shape: {X_train_tr.shape}")
    print(f"Transformed test shape: {X_test_tr.shape}")
    
    # === SAVE EVERYTHING ===
    
    # Save preprocessed data
    out_path = os.path.join(WORKING_DIR, "preprocessed.pkl")
    with open(out_path, "wb") as f:
        pickle.dump((X_train_tr, X_test_tr, y_train.values, y_test.values), f)
    
    # Save transformer for production use (BONUS)
    transformer_path = os.path.join(MODEL_DIR, "transformer.pkl")
    with open(transformer_path, "wb") as f:
        pickle.dump(ct, f)
    
    print(f"\n✓ Preprocessing complete. Saved to {out_path}")
    return out_path


def separate_data_outputs(file_path: str) -> str:
    """
    Passthrough; kept so the DAG composes cleanly.
    """
    return file_path


def build_model(file_path: str, filename: str) -> str:
    """
    Train multiple models and save the best one.
    HR-specific: Focus on recall for predicting attrition.
    """
    with open(file_path, "rb") as f:
        X_train, X_test, y_train, y_test = pickle.load(f)
    
    print("\n=== Training Multiple Models ===")
    
    # Define models to compare
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            class_weight='balanced',  # Handle class imbalance
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
    }
    
    best_model = None
    best_score = 0
    best_name = None
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Evaluate on test set
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        print(f"  Train Accuracy: {train_score:.4f}")
        print(f"  Test Accuracy:  {test_score:.4f}")
        
        # Select best based on test accuracy
        if test_score > best_score:
            best_score = test_score
            best_model = model
            best_name = name
    
    print(f"\n🏆 Best Model: {best_name} (Accuracy: {best_score:.4f})")
    
    # Save best model
    model_path = os.path.join(MODEL_DIR, filename)
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    
    # Save model metadata
    metadata = {
        'model_name': best_name,
        'test_accuracy': float(best_score),
        'training_date': pd.Timestamp.now().isoformat()
    }
    
    metadata_path = os.path.join(MODEL_DIR, "model_metadata.pkl")
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    
    print(f"✓ Model saved to {model_path}")
    return model_path


def load_model(file_path: str, filename: str) -> dict:
    """
    Load model, evaluate performance, and return detailed metrics.
    """
    with open(file_path, "rb") as f:
        X_train, X_test, y_train, y_test = pickle.load(f)
    
    model_path = os.path.join(MODEL_DIR, filename)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    print("\n=== Model Evaluation ===")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    test_accuracy = model.score(X_test, y_test)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Stayed', 'Left']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Save detailed metrics
    metrics = {
        'test_accuracy': float(test_accuracy),
        'roc_auc': float(roc_auc),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': cm.tolist(),
        'evaluation_date': pd.Timestamp.now().isoformat()
    }
    
    metrics_path = os.path.join(MODEL_DIR, "evaluation_metrics.pkl")
    with open(metrics_path, "wb") as f:
        pickle.dump(metrics, f)
    
    print(f"\n✓ Metrics saved to {metrics_path}")
    
    # Return for potential downstream use
    return metrics