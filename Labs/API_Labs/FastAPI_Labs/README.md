# FastAPI Lab — California Housing Price Prediction

An extension of the FastAPI Lab assignment using the **California Housing dataset** instead of Iris, exploring additional FastAPI concepts including multiple endpoints, async functions, batch prediction, path parameters, and Pydantic validation.

---

## What I Implemented

### Beyond the Base Lab (Iris Dataset)

| Concept | Iris Lab | This Implementation |
|---|---|---|
| Dataset | Iris classification | California Housing regression |
| Model | Decision Tree | Random Forest Regressor |
| Endpoints | 1 (`/predict`) | 5 (`/predict`, `/predict-batch`, `/categories/{category}`, `/features`, `/health`) |
| Async | No | Yes — all endpoints use `async def` |
| Batch prediction | No | Yes — predict up to 50 houses in one request |
| Path parameters | No | Yes — `GET /categories/{category}` |
| Custom Pydantic validators | No | Yes — range checks on MedInc, HouseAge, AveRooms |
| HTTPException | No | Yes — 404 for invalid category, 500 for prediction errors |

---

## Project Structure

```
fastapi_lab/
├── src/
│   ├── main.py        ← FastAPI app + all 5 endpoints
│   ├── data.py        ← Pydantic request/response models
│   ├── predict.py     ← Model loading + prediction logic
│   └── train.py       ← Train and save model
├── model/
│   └── housing_model.pkl
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# Create and activate virtual environment
python -m venv fastapi_lab_env
source fastapi_lab_env/bin/activate    # Mac/Linux
fastapi_lab_env\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Lab

**Step 1 — Train the model:**
```bash
cd src
python train.py
```
Output:
```
Features : ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
Samples  : 20640
RMSE : 0.5057
R2   : 0.8049
Saved → model/housing_model.pkl
```

**Step 2 — Start the API:**
```bash
uvicorn main:app --reload
```

**Step 3 — Test endpoints:**

Visit `http://127.0.0.1:8000/docs` for interactive Swagger UI.

---

## API Endpoints

### POST `/predict` — Single house prediction

```json
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
```

Response:
```json
{
  "predicted_price": 4.152,
  "price_category": "expensive"
}
```

---

### POST `/predict-batch` — Batch prediction (up to 50 houses)

```json
{
  "houses": [
    {
      "MedInc": 8.3252, "HouseAge": 41.0, "AveRooms": 6.984,
      "AveBedrms": 1.024, "Population": 322.0, "AveOccup": 2.555,
      "Latitude": 37.88, "Longitude": -122.23
    },
    {
      "MedInc": 3.5, "HouseAge": 20.0, "AveRooms": 4.5,
      "AveBedrms": 1.1, "Population": 800.0, "AveOccup": 2.8,
      "Latitude": 37.5, "Longitude": -122.0
    }
  ]
}
```

Response:
```json
{
  "predictions": [4.152, 2.341],
  "count": 2,
  "average_price": 3.2465,
  "min_price": 2.341,
  "max_price": 4.152
}
```

---

### GET `/categories/{category}` — Price category info

```
GET /categories/affordable
GET /categories/mid-range
GET /categories/expensive
```

Response:
```json
{
  "label": "Affordable",
  "range": "Under $150,000",
  "threshold": "predicted_price < 1.5"
}
```

Returns **404** for invalid category:
```json
{
  "detail": "Category 'luxury' not found. Valid categories: ['affordable', 'mid-range', 'expensive']"
}
```

---

### GET `/features` — Feature descriptions

Returns all 8 feature names and descriptions.

---

### GET `/health` — Health check

```json
{ "status": "healthy" }
```

---

## Key Concepts Demonstrated

### 1. Async Endpoints
All endpoints use `async def` — enables non-blocking I/O so the server handles multiple concurrent requests without waiting for each to finish.

```python
@app.post("/predict", response_model=HousingResponse)
async def predict(data: HousingData):
    ...
```

### 2. Pydantic Validation with Custom Validators
Beyond basic type checking — custom `@validator` rules raise 422 errors automatically before the prediction runs.

```python
@validator("MedInc")
def income_must_be_positive(cls, v):
    if v < 0:
        raise ValueError("MedInc must be positive")
    return v
```

### 3. Batch Prediction
Single model call for multiple houses — more efficient than looping `/predict`.

```python
def predict_batch(all_features: list) -> list:
    features_array = np.array(all_features)
    predictions = model.predict(features_array)
    return [round(float(p), 4) for p in predictions]
```

### 4. Path Parameters + HTTPException
Dynamic URL segments with proper error handling.

```python
@app.get("/categories/{category}")
async def get_category_info(category: str):
    if category not in category_info:
        raise HTTPException(status_code=404, detail=f"Category '{category}' not found")
```

### 5. Model loaded once at startup
Model loaded when server starts — not on every request — avoiding repeated disk reads.

```python
with open("../model/housing_model.pkl", "rb") as f:
    model = pickle.load(f)
```

---

## Testing Validation Errors

Send invalid data to see Pydantic return **422 Unprocessable Entity**:

```json
{
  "MedInc": -5,
  "HouseAge": 200,
  "AveRooms": 0,
  "AveBedrms": 1.0,
  "Population": 500.0,
  "AveOccup": 2.5,
  "Latitude": 37.5,
  "Longitude": -122.0
}
```

Expected response:
```json
{
  "detail": [
    {"loc": ["body", "MedInc"], "msg": "MedInc must be positive"},
    {"loc": ["body", "HouseAge"], "msg": "HouseAge must be between 0 and 100"},
    {"loc": ["body", "AveRooms"], "msg": "AveRooms must be greater than 0"}
  ]
}
```