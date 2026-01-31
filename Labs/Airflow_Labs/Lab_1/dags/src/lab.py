import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
import pickle
import os
import base64

SPOTIFY_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]

BASE_DIR = os.path.dirname(__file__)                 # dags/src
DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "../data"))   # dags/data
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "../../model"))  # project_root/model
SCALER_FILENAME = "scaler.pkl"

def load_data():
    """
    Loads data from a CSV file, serializes it, and returns the serialized data.
    Returns:
        str: Base64-encoded serialized data (JSON-safe).
    """
    print("We are here")
    df = pd.read_csv(os.path.join(DATA_PATH, "data.csv"))
    serialized_data = pickle.dumps(df)                    # bytes
    return base64.b64encode(serialized_data).decode("ascii")  # JSON-safe string

def feature_engineering(data_b64: str):
    """
    NEW STEP:
      - Keep only Spotify audio features
      - Coerce to numeric + drop missing
      - Clip outliers (1% to 99%) so KMeans isn't dominated by extreme values
      - Add engineered features:
          mood = (valence + energy)/2
          intensity = energy - acousticness
      - Optional sampling for faster Airflow/Docker runs

    Returns:
        str: Base64-encoded pickled DataFrame (JSON-safe).
    """
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    missing = [c for c in SPOTIFY_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Spotify dataset missing required columns: {missing}")

    clustering_data = df[SPOTIFY_FEATURES].copy()

    for col in SPOTIFY_FEATURES:
        clustering_data[col] = pd.to_numeric(clustering_data[col], errors="coerce")
    clustering_data = clustering_data.dropna()

    # Outlier clipping (winsorize)
    for col in SPOTIFY_FEATURES:
        lo = clustering_data[col].quantile(0.01)
        hi = clustering_data[col].quantile(0.99)
        clustering_data[col] = clustering_data[col].clip(lo, hi)

    # Feature engineering
    clustering_data["mood"] = (clustering_data["valence"] + clustering_data["energy"]) / 2.0
    clustering_data["intensity"] = clustering_data["energy"] - clustering_data["acousticness"]

    # Optional sampling for speed
    if len(clustering_data) > 20000:
        clustering_data = clustering_data.sample(n=20000, random_state=42)

    serialized_data = pickle.dumps(clustering_data)
    return base64.b64encode(serialized_data).decode("ascii")

def data_preprocessing(data_b64: str):
    """
    Takes engineered Spotify features (including mood/intensity),
    fits MinMaxScaler on TRAIN data, saves scaler, returns scaled matrix as b64.
    """
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    clustering_data = df.copy()

    min_max_scaler = MinMaxScaler()
    clustering_data_minmax = min_max_scaler.fit_transform(clustering_data)

    # save scaler so test data uses the same scaling
    os.makedirs(MODEL_PATH, exist_ok=True)
    scaler_path = os.path.join(MODEL_PATH, SCALER_FILENAME)
    with open(scaler_path, "wb") as f:
        pickle.dump(min_max_scaler, f)

    clustering_serialized_data = pickle.dumps(clustering_data_minmax)
    return base64.b64encode(clustering_serialized_data).decode("ascii")


def build_save_model(data_b64: str, filename: str):
    """
    Tries k=2..20, computes SSE, finds elbow, refits KMeans on elbow-k,
    saves that model, returns SSE list.
    """
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    kmeans_kwargs = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42}
    sse = []
    k_range = range(2, 21)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(df)
        sse.append(kmeans.inertia_)

    kl = KneeLocator(list(k_range), sse, curve="convex", direction="decreasing")
    best_k = kl.elbow

    # fallback if KneeLocator can't find an elbow
    if best_k is None:
        best_k = 5

    final_model = KMeans(n_clusters=int(best_k), **kmeans_kwargs)
    final_model.fit(df)

    os.makedirs(MODEL_PATH, exist_ok=True)
    output_path = os.path.join(MODEL_PATH, filename)
    with open(output_path, "wb") as f:
        pickle.dump(final_model, f)

    print(f"Selected k (elbow): {best_k}")
    return sse


def load_model_elbow(filename: str, sse: list):
    """
    Loads the saved model and reports elbow-k.
    Predicts the cluster for the first valid row in test.csv.
    """
    output_path = os.path.join(MODEL_PATH, filename)
    loaded_model = pickle.load(open(output_path, "rb"))

    kl = KneeLocator(range(2, 21), sse, curve="convex", direction="decreasing")
    print(f"Optimal no. of clusters: {kl.elbow}")

    # load test.csv once (local path)
    df = pd.read_csv(os.path.join(DATA_PATH, "test.csv"))
    print("\n" + "="*60)
    print("TEST DATA PREDICTION")
    print("="*60)
    print("\nOriginal test data:")
    print(df.to_string(index=False))

    missing = [c for c in SPOTIFY_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"test.csv missing required Spotify columns: {missing}")

    clustering_data = df[SPOTIFY_FEATURES].copy()

    for col in SPOTIFY_FEATURES:
        clustering_data[col] = pd.to_numeric(clustering_data[col], errors="coerce")
    clustering_data = clustering_data.dropna()

    if len(clustering_data) == 0:
        raise ValueError("test.csv has no valid rows after numeric conversion + dropna")

    # match feature_engineering(): clip outliers
    for col in SPOTIFY_FEATURES:
        lo = clustering_data[col].quantile(0.01)
        hi = clustering_data[col].quantile(0.99)
        clustering_data[col] = clustering_data[col].clip(lo, hi)

    # add engineered features to match training
    clustering_data["mood"] = (clustering_data["valence"] + clustering_data["energy"]) / 2.0
    clustering_data["intensity"] = clustering_data["energy"] - clustering_data["acousticness"]

    print("\nAfter feature engineering:")
    print(f"  mood = {clustering_data['mood'].iloc[0]:.3f}")
    print(f"  intensity = {clustering_data['intensity'].iloc[0]:.3f}")

    # load training scaler and transform test data
    scaler_path = os.path.join(MODEL_PATH, SCALER_FILENAME)
    min_max_scaler = pickle.load(open(scaler_path, "rb"))
    clustering_data_minmax = min_max_scaler.transform(clustering_data)

    pred = loaded_model.predict(clustering_data_minmax)[0]

    print(f"\n🎯 PREDICTED CLUSTER: {pred}")
    print("\nThis song belongs to Cluster {}, which means it has similar".format(pred))
    print("audio characteristics to other songs in that cluster.")
    print("="*60 + "\n")

    try:
        return int(pred)
    except Exception:
        return pred.item() if hasattr(pred, "item") else pred


def visualize_clusters(filename: str, sse: list):
    """
    Visualizes the clustering results:
    1. Elbow plot showing SSE vs K
    2. Cluster distribution
    3. 2D scatter plot of clusters (using PCA)
    4. Cluster characteristics heatmap
    """
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for Airflow
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.decomposition import PCA
    
    # Load the saved model
    output_path = os.path.join(MODEL_PATH, filename)
    loaded_model = pickle.load(open(output_path, "rb"))
    
    # Load and preprocess the training data (same as in build_save_model)
    df = pd.read_csv(os.path.join(DATA_PATH, "data.csv"))
    
    missing = [c for c in SPOTIFY_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    
    clustering_data = df[SPOTIFY_FEATURES].copy()
    
    for col in SPOTIFY_FEATURES:
        clustering_data[col] = pd.to_numeric(clustering_data[col], errors="coerce")
    clustering_data = clustering_data.dropna()
    
    # Clip outliers
    for col in SPOTIFY_FEATURES:
        lo = clustering_data[col].quantile(0.01)
        hi = clustering_data[col].quantile(0.99)
        clustering_data[col] = clustering_data[col].clip(lo, hi)
    
    # Add engineered features
    clustering_data["mood"] = (clustering_data["valence"] + clustering_data["energy"]) / 2.0
    clustering_data["intensity"] = clustering_data["energy"] - clustering_data["acousticness"]
    
    # Optional sampling
    if len(clustering_data) > 20000:
        clustering_data = clustering_data.sample(n=20000, random_state=42)
    
    # Load scaler and transform
    scaler_path = os.path.join(MODEL_PATH, SCALER_FILENAME)
    min_max_scaler = pickle.load(open(scaler_path, "rb"))
    clustering_data_minmax = min_max_scaler.transform(clustering_data)
    
    # Get cluster predictions
    clusters = loaded_model.predict(clustering_data_minmax)
    
    # Create output directory for plots
    plots_dir = os.path.abspath(os.path.join(BASE_DIR, "../../plots"))
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Elbow Plot
    plt.figure(figsize=(10, 6))
    k_range = range(2, 21)
    plt.plot(k_range, sse, 'bo-', linewidth=2, markersize=8)
    
    # Mark the elbow
    kl = KneeLocator(list(k_range), sse, curve="convex", direction="decreasing")
    if kl.elbow:
        plt.axvline(x=kl.elbow, color='r', linestyle='--', linewidth=2, label=f'Elbow at k={kl.elbow}')
        plt.plot(kl.elbow, sse[kl.elbow-2], 'ro', markersize=12)
    
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Sum of Squared Errors (SSE)', fontsize=12)
    plt.title('Elbow Method For Optimal k', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    elbow_plot_path = os.path.join(plots_dir, "elbow_plot.png")
    plt.savefig(elbow_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved elbow plot to: {elbow_plot_path}")
    
    # 2. Cluster Distribution
    plt.figure(figsize=(10, 6))
    cluster_counts = pd.Series(clusters).value_counts().sort_index()
    colors = plt.cm.viridis(range(len(cluster_counts)))
    plt.bar(cluster_counts.index, cluster_counts.values, color=colors, edgecolor='black', linewidth=1.5)
    plt.xlabel('Cluster', fontsize=12)
    plt.ylabel('Number of Songs', fontsize=12)
    plt.title('Distribution of Songs Across Clusters', fontsize=14, fontweight='bold')
    plt.xticks(cluster_counts.index)
    plt.grid(axis='y', alpha=0.3)
    
    # Add count labels on bars
    for i, v in enumerate(cluster_counts.values):
        plt.text(cluster_counts.index[i], v + max(cluster_counts.values)*0.01, 
                str(v), ha='center', va='bottom', fontweight='bold')
    
    distribution_plot_path = os.path.join(plots_dir, "cluster_distribution.png")
    plt.savefig(distribution_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved cluster distribution to: {distribution_plot_path}")
    
    # 3. 2D PCA Visualization
    pca = PCA(n_components=2, random_state=42)
    principal_components = pca.fit_transform(clustering_data_minmax)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(principal_components[:, 0], principal_components[:, 1], 
                         c=clusters, cmap='viridis', alpha=0.6, edgecolors='w', s=50, linewidth=0.5)
    
    # Add cluster centers
    centers_pca = pca.transform(loaded_model.cluster_centers_)
    plt.scatter(centers_pca[:, 0], centers_pca[:, 1], 
               c='red', marker='X', s=300, edgecolors='black', linewidth=2, label='Centroids')
    
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    plt.title('Clusters Visualized in 2D (PCA)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    pca_plot_path = os.path.join(plots_dir, "clusters_pca.png")
    plt.savefig(pca_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved PCA plot to: {pca_plot_path}")
    
    # 4. Cluster Characteristics (mean values for each feature)
    clustering_data_with_clusters = clustering_data.copy()
    clustering_data_with_clusters['cluster'] = clusters
    cluster_means = clustering_data_with_clusters.groupby('cluster').mean()
    
    # Plot heatmap of cluster characteristics
    plt.figure(figsize=(14, 8))
    sns.heatmap(cluster_means.T, annot=True, fmt='.2f', cmap='YlOrRd', 
                cbar_kws={'label': 'Mean Value'}, linewidths=0.5)
    plt.title('Mean Feature Values per Cluster', fontsize=14, fontweight='bold')
    plt.xlabel('Cluster', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.tight_layout()
    
    heatmap_path = os.path.join(plots_dir, "cluster_characteristics.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved cluster characteristics to: {heatmap_path}")
    
    # Print cluster statistics
    print("\n" + "="*60)
    print("CLUSTER ANALYSIS SUMMARY")
    print("="*60)
    print(f"\nTotal number of clusters: {loaded_model.n_clusters}")
    print(f"Total songs clustered: {len(clusters)}")
    print("\nCluster sizes:")
    for cluster_id, count in cluster_counts.items():
        percentage = (count / len(clusters)) * 100
        print(f"  Cluster {cluster_id}: {count} songs ({percentage:.1f}%)")
    
    print("\nTop 3 defining features per cluster:")
    for cluster_id in cluster_means.index:
        top_features = cluster_means.loc[cluster_id].nlargest(3)
        print(f"\n  Cluster {cluster_id}:")
        for feature, value in top_features.items():
            print(f"    - {feature}: {value:.3f}")
    
    print("\n" + "="*60)
    print(f"All plots saved to: {plots_dir}")
    print("="*60 + "\n")
    
    return {
        'n_clusters': int(loaded_model.n_clusters),
        'cluster_sizes': {int(k): int(v) for k, v in cluster_counts.items()},
        'plots_directory': plots_dir
    }