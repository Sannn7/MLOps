"""
src/data.py

Pydantic models for request and response validation.

FastAPI uses these to:
1. Validate incoming request data
2. Return 422 if data is wrong type or missing
3. Auto-generate API documentation
4. Serialize response data to JSON

Features (from sklearn's fetch_california_housing):
    MedInc      - median income in block group
    HouseAge    - median house age in block group
    AveRooms    - average number of rooms per household
    AveBedrms   - average number of bedrooms per household
    Population  - block group population
    AveOccup    - average number of household members
    Latitude    - block group latitude
    Longitude   - block group longitude
"""

from pydantic import BaseModel, validator


# ─────────────────────────────────────────────────────────────
# REQUEST MODELS
# ─────────────────────────────────────────────────────────────

class HousingData(BaseModel):
    """
    Input features for single house price prediction.
    FastAPI validates types and raises 422 if wrong.
    """
    MedInc:     float
    HouseAge:   float
    AveRooms:   float
    AveBedrms:  float
    Population: float
    AveOccup:   float
    Latitude:   float
    Longitude:  float

    @validator("MedInc")
    def income_must_be_positive(cls, v):
        if v < 0:
            raise ValueError("MedInc must be positive")
        return v

    @validator("HouseAge")
    def age_must_be_valid(cls, v):
        if v < 0 or v > 100:
            raise ValueError("HouseAge must be between 0 and 100")
        return v

    @validator("AveRooms")
    def rooms_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("AveRooms must be greater than 0")
        return v


class BatchHousingData(BaseModel):
    """
    Input for batch prediction — list of houses.
    Maximum 50 houses per request.
    """
    houses: list[HousingData]

    @validator("houses")
    def must_have_at_least_one(cls, v):
        if len(v) == 0:
            raise ValueError("Must provide at least one house")
        if len(v) > 50:
            raise ValueError("Maximum 50 houses per batch request")
        return v


# ─────────────────────────────────────────────────────────────
# RESPONSE MODELS
# ─────────────────────────────────────────────────────────────

class HousingResponse(BaseModel):
    """
    Response for single house prediction.
    predicted_price is in hundreds of thousands USD.
    """
    predicted_price: float
    price_category:  str    # "affordable", "mid-range", "expensive"


class BatchHousingResponse(BaseModel):
    """
    Response for batch prediction.
    Returns all predictions + summary statistics.
    """
    predictions:   list[float]
    count:         int
    average_price: float
    min_price:     float
    max_price:     float