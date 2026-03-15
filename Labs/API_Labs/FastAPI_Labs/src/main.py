"""
src/main.py

California Housing Price Prediction API using FastAPI.

New concepts explored vs Iris lab:
1. Multiple endpoints (/predict, /predict-batch, /health,
                       /features, /categories/{category})
2. Async endpoints (async def)
3. HTTPException for error handling
4. Path parameters (GET /categories/{category})

Run:
    uvicorn main:app --reload
"""

from fastapi import FastAPI, HTTPException
from data import (
    HousingData, HousingResponse,
    BatchHousingData, BatchHousingResponse
)
from predict import predict_price, predict_batch, get_price_category

# ── App instance ──
app = FastAPI(
    title="California Housing Price Predictor",
    description="Predict house prices using Random Forest Regressor",
    version="1.0.0"
)


# ─────────────────────────────────────────────────────────────
# ENDPOINT 1 — Single prediction
# Same concept as Iris lab /predict endpoint
# Added: async def + HTTPException + price category
# ─────────────────────────────────────────────────────────────
@app.post("/predict", response_model=HousingResponse)
async def predict(data: HousingData):
    """
    Predict price for a single house.

    Example request body:
    {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.984,
        "AveBedrms": 1.024,
        "Population": 322.0,
        "AveOccup": 2.555,
        "Latitude": 37.88,
        "Longitude": -122.23
    }
    """
    try:
        price = predict_price(
            MedInc=data.MedInc,
            HouseAge=data.HouseAge,
            AveRooms=data.AveRooms,
            AveBedrms=data.AveBedrms,
            Population=data.Population,
            AveOccup=data.AveOccup,
            Latitude=data.Latitude,
            Longitude=data.Longitude
        )
        category = get_price_category(price)

        return HousingResponse(
            predicted_price=price,
            price_category=category
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# ─────────────────────────────────────────────────────────────
# ENDPOINT 2 — Batch prediction
# New concept: predict multiple houses in one request
# More efficient than calling /predict in a loop
# ─────────────────────────────────────────────────────────────
@app.post("/predict-batch", response_model=BatchHousingResponse)
async def predict_batch_endpoint(data: BatchHousingData):
    """
    Predict house prices for multiple houses at once.
    More efficient than calling /predict in a loop.
    Maximum 50 houses per request.

    Example request:
    {
        "houses": [
            {
                "MedInc": 8.3252, "HouseAge": 41.0,
                "AveRooms": 6.984, "AveBedrms": 1.024,
                "Population": 322.0, "AveOccup": 2.555,
                "Latitude": 37.88, "Longitude": -122.23
            },
            {
                "MedInc": 5.1, "HouseAge": 25.0,
                "AveRooms": 5.0, "AveBedrms": 1.1,
                "Population": 500.0, "AveOccup": 2.8,
                "Latitude": 37.5, "Longitude": -122.0
            }
        ]
    }
    """
    try:
        # Extract features for all houses as list of lists
        all_features = [
            [
                h.MedInc, h.HouseAge, h.AveRooms,
                h.AveBedrms, h.Population, h.AveOccup,
                h.Latitude, h.Longitude
            ]
            for h in data.houses
        ]

        # Run batch prediction — one model call for all houses
        predictions = predict_batch(all_features)

        return BatchHousingResponse(
            predictions=predictions,
            count=len(predictions),
            average_price=round(sum(predictions) / len(predictions), 4),
            min_price=round(min(predictions), 4),
            max_price=round(max(predictions), 4)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


# ─────────────────────────────────────────────────────────────
# ENDPOINT 3 — Category info (path parameter)
# New concept: GET with dynamic path parameter
# HTTPException 404 for invalid category
# ─────────────────────────────────────────────────────────────
@app.get("/categories/{category}")
async def get_category_info(category: str):
    """
    Get price range info for a category.
    Valid: affordable, mid-range, expensive

    Example: GET /categories/affordable
    """
    category_info = {
        "affordable": {
            "label": "Affordable",
            "range": "Under $150,000",
            "threshold": "predicted_price < 1.5"
        },
        "mid-range": {
            "label": "Mid-Range",
            "range": "$150,000 - $300,000",
            "threshold": "1.5 <= predicted_price <= 3.0"
        },
        "expensive": {
            "label": "Expensive",
            "range": "Over $300,000",
            "threshold": "predicted_price > 3.0"
        }
    }

    if category not in category_info:
        raise HTTPException(
            status_code=404,
            detail=f"Category '{category}' not found. "
                   f"Valid categories: {list(category_info.keys())}"
        )

    return category_info[category]


# ─────────────────────────────────────────────────────────────
# ENDPOINT 4 — Feature descriptions
# New concept: GET metadata endpoint
# ─────────────────────────────────────────────────────────────
@app.get("/features")
async def get_features():
    """
    Returns feature names and descriptions.
    """
    return {
        "features": [
            {"name": "MedInc",     "description": "Median income (tens of thousands USD)"},
            {"name": "HouseAge",   "description": "Median house age (years)"},
            {"name": "AveRooms",   "description": "Average rooms per household"},
            {"name": "AveBedrms",  "description": "Average bedrooms per household"},
            {"name": "Population", "description": "Block group population"},
            {"name": "AveOccup",   "description": "Average household members"},
            {"name": "Latitude",   "description": "Block group latitude"},
            {"name": "Longitude",  "description": "Block group longitude"},
        ]
    }


# ─────────────────────────────────────────────────────────────
# ENDPOINT 5 — Health check
# New concept: monitoring endpoint
# ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """
    Health check — returns 200 if API is running.
    """
    return {"status": "healthy"}