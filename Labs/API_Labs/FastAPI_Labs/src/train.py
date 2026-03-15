

import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing  

# ── Load Dataset ──
data = fetch_california_housing()
X, y = data.data, data.target

feature_names = data.feature_names
print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Features: {feature_names}")

# ── Train/Test Split ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Train Model ──
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# ── Evaluate ──
y_pred = model.predict(X_test)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"  RMSE : {rmse:.4f}")
print(f"  R²   : {r2:.4f}")

# ── Save Model ──
with open("../model/housing_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved to model/housing_model.pkl")
print("Now run: uvicorn main:app --reload")