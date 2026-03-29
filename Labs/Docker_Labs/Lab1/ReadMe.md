# Docker Lab 1 - Titanic Survival Prediction

## Overview
This lab demonstrates core Docker concepts by containerizing a machine learning 
pipeline that predicts Titanic passenger survival using a Gradient Boosting Classifier.
The dataset is loaded directly from a URL, preprocessed, trained, evaluated, 
and the model is saved using Docker volume mounting for persistence.

## What I Implemented

### Machine Learning Pipeline
- Loaded Titanic dataset from URL using pandas
- Preprocessed features (handled missing values, encoded categorical variables)
- Trained a Gradient Boosting Classifier
- Evaluated model with accuracy score and classification report
- Saved trained model as `titanic_model.pkl`

### Docker Concepts Applied
1. **Slim Base Image** — used `python:3.10-slim` to reduce image size
2. **Image Labels** — added maintainer, version, and description metadata
3. **Environment Variables** — configured Python behavior via `ENV`
4. **Layer Caching Optimization** — copied requirements before source code to cache dependencies
5. **Volume Mounting** — persisted trained model outside the container
6. **Non-root User Security** — container runs as `appuser` instead of root
7. **Health Check** — Docker monitors container health automatically

## Results
```
Model Accuracy: 0.7832

Classification Report:
              precision    recall  f1-score   support
           0       0.83      0.82      0.82        87
           1       0.72      0.73      0.73        56
    accuracy                           0.78       143
   macro avg       0.77      0.77      0.77       143
weighted avg       0.78      0.78      0.78       143

Model saved to titanic_model.pkl
```

## How to Run the Lab

### Prerequisites
- Docker Desktop installed and running
- Git installed

### Step 1 — Clone the repository
```bash
git clone https://github.com/Sannn7/MLOps.git
cd MLOps/Labs/Docker_Labs/Lab1
```

### Step 2 — Build the Docker image
```bash
docker build -t lab1:v2 .
```

### Step 3 — Run the container with volume mount
```bash
docker run -v $(pwd)/models:/app/models lab1:v2
```

### Step 4 — Verify model was saved outside container
```bash
ls models/
# titanic_model.pkl should be present
```

### Step 5 — Save image to tar file
```bash
docker save lab1:v2 > my_image.tar
```

### Step 6 — Inspect image details and layers
```bash
docker inspect lab1:v2
docker images lab1:v2
```

### Step 7 — View container logs
```bash
docker ps -a
docker logs <container_id>
```

### Step 8 — Clean up
```bash
docker stop $(docker ps -q)
docker rm $(docker ps -a -q)
```

## Key Takeaways
- Docker ensures the pipeline runs consistently across any environment
- Volume mounting allows model persistence even after the container stops
- Non-root user and health checks are production-grade best practices
- Layer caching significantly speeds up rebuilds during development