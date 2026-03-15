"""
src/predict.py

Loads the trained model and runs predictions.
Called by main.py endpoints.
"""

import pickle
import numpy as np

# Load model once at startup
# Not inside the function — avoids reloading on every request
with open("../model/housing_model.pkl", "rb") as f:
    model = pickle.load(f)

print("Model loaded successfully")


def predict_price(
    MedInc: float,
    HouseAge: float,
    AveRooms: float,
    AveBedrms: float,
    Population: float,
    AveOccup: float,
    Latitude: float,
    Longitude: float
) -> float:
    """
    Run prediction for one house.
    Returns predicted price in hundreds of thousands USD.
    """
    features = np.array([[
        MedInc, HouseAge, AveRooms, AveBedrms,
        Population, AveOccup, Latitude, Longitude
    ]])
    prediction = model.predict(features)
    return round(float(prediction[0]), 4)


def predict_batch(all_features: list) -> list:
    """
    Run prediction for multiple houses at once.
    all_features is a list of lists — each inner list
    is one house's features in order:
    [MedInc, HouseAge, AveRooms, AveBedrms,
     Population, AveOccup, Latitude, Longitude]

    Returns list of predicted prices.
    More efficient than calling predict_price() in a loop
    because model.predict() handles all rows in one call.
    """
    features_array = np.array(all_features)
    predictions = model.predict(features_array)
    return [round(float(p), 4) for p in predictions]


def get_price_category(price: float) -> str:
    """
    Categorize predicted price into buckets.
    price is in hundreds of thousands USD.

    < 1.5  → affordable   (under $150K)
    1.5-3  → mid-range    ($150K - $300K)
    > 3    → expensive    (over $300K)
    """
    if price < 1.5:
        return "affordable"
    elif price <= 3.0:
        return "mid-range"
    return "expensive"