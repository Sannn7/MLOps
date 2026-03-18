# Lab 2 - Serverless ML Pipeline with Google Cloud Functions

## Overview
This lab implements a serverless machine learning pipeline using Google Cloud Functions, Pub/Sub, and Workflows. Modified from original to use Titanic survival dataset and Random Forest classifier.

## Modifications from Original Lab
- Dataset: Titanic survival dataset (5000 rows) instead of Iris
- Target column: Survived (binary classification) instead of species
- Functions use HTTP triggers instead of GCS event triggers
- workflow.yaml updated with project ID, bucket name, and Titanic feature inputs

## Architecture
HTTP Request -> process_data -> Pub/Sub -> train_model -> GCS -> ml_model_predict -> prediction

## Results
- Workflow State: SUCCEEDED
- Duration: 39.87 seconds
- Output: {"prediction": [1]} - passenger survived

## curl Test
curl -X POST https://us-central1-mlops-vertexai-project.cloudfunctions.net/ml_model_predict \
    -H "Content-Type: application/json" \
    -d '{"features": [3, 1, 25.0, 0, 0, 7.25, 2]}'

Output: {"prediction": [1]}
