# Airflow Lab 1: Spotify Song Clustering Pipeline
**Student:** Sanika Killekar  
**Date:** January 31, 2026

## Overview
This project implements an end-to-end ML pipeline using Apache Airflow to cluster Spotify songs based on audio features using K-Means clustering with automated elbow method optimization.

## Project Structure
```
Lab_1/
├── dags/
│   ├── data/
│   │   ├── data.csv          # Training data
│   │   └── test.csv          # Test data
│   ├── src/
│   │   └── lab.py            # Core ML functions
│   └── spotify_dag.py        # Airflow DAG definition
├── model/                    # Saved models (generated)
│   ├── model.sav
│   └── scaler.pkl
├── plots/                    # Visualizations (generated)
│   ├── elbow_plot.png
│   ├── cluster_distribution.png
│   ├── clusters_pca.png
│   └── cluster_characteristics.png
├── logs/                     # Airflow logs
├── airflow.cfg              # Airflow configuration
└── README.md                # This file
```

## Pipeline Architecture

The DAG consists of 6 tasks executed sequentially:

1. **load_data_task**: Loads training data from CSV
2. **feature_engineering_task**: 
   - Clips outliers (1%-99% quantiles)
   - Creates engineered features: `mood = (valence + energy)/2`, `intensity = energy - acousticness`
3. **data_preprocessing_task**: Applies MinMax scaling and saves scaler
4. **build_save_model_task**: 
   - Tests K=2 to K=20
   - Uses elbow method to find optimal K
   - Trains final KMeans model
5. **load_model_task**: Loads model and predicts cluster for test data
6. **visualize_clusters_task**: Generates 4 visualization plots

## Setup Instructions

### Prerequisites
- Python 3.11 or 3.12
- Virtual environment

### Installation
```bash
# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install apache-airflow pandas numpy scikit-learn kneed matplotlib seaborn

# Initialize Airflow database
airflow db init

# Migrate database schema
airflow db migrate
```

## Running the Pipeline
```bash
# Activate virtual environment
source .venv/bin/activate

# Run the DAG
airflow dags test Airflow_Lab1
```

**Expected runtime:** ~15-30 seconds

## Results Location

### 1. Visualizations (Primary Results)
All plots are saved in: `plots/`

- **elbow_plot.png**: Shows optimal K=8 was selected
- **cluster_distribution.png**: Distribution of songs across 8 clusters
- **clusters_pca.png**: 2D PCA visualization of clusters with centroids
- **cluster_characteristics.png**: Heatmap showing mean feature values per cluster

### 2. Test Data Prediction
Terminal output shows:
```
============================================================
TEST DATA PREDICTION
============================================================
Original test data:
[Feature values displayed]

After feature engineering:
  mood = 0.745
  intensity = 0.720

🎯 PREDICTED CLUSTER: 3
============================================================
```

### 3. Saved Models
Located in: `model/`
- `model.sav`: Trained KMeans model (K=8)
- `scaler.pkl`: MinMaxScaler fitted on training data

## Key Results

### Clustering Results
- **Optimal number of clusters:** 8 (determined by elbow method)
- **Total songs clustered:** [varies based on data.csv size]
- **Algorithm:** KMeans with random initialization

### Test Song Prediction
The test song with features:
- Danceability: 0.72
- Energy: 0.84
- Valence: 0.65
- Tempo: 128 BPM
- **Mood:** 0.745 (engineered)
- **Intensity:** 0.720 (engineered)

Was classified into **Cluster 3**, indicating it belongs to an upbeat, high-energy, danceable song category.

## Technical Implementation Details

### Feature Engineering
- Outlier clipping using 1st and 99th percentiles
- Created composite features: mood and intensity
- Optional sampling (20,000 songs max) for performance

### Data Preprocessing
- MinMaxScaler normalization
- Scaler persistence for consistent test data transformation

### Model Selection
- Automated elbow method using KneeLocator library
- SSE computed for K=2 to K=20
- Selected K=8 as optimal

### Visualization
- PCA reduction to 2D for cluster visualization
- Cluster centroids marked on scatter plot
- Heatmap showing cluster characteristics across all features

## Dependencies
```
apache-airflow==2.10.4
pandas
numpy
scikit-learn
kneed
matplotlib
seaborn
```

## Troubleshooting

### Common Issues
1. **Python 3.14 compatibility error**: Use Python 3.11 or 3.12
2. **Database schema error**: Run `airflow db migrate`
3. **Missing packages**: Install all dependencies listed above

## Contact
Sanika Killekar  
Northeastern University  
MS Data Science