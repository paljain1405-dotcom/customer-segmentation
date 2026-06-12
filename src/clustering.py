import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import json


def run_kmeans_clustering(rfm_df, n_clusters=4):
    features = ["Recency", "Frequency", "Monetary"]
    X = rfm_df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm_df = rfm_df.copy()
    rfm_df["Cluster"] = kmeans.fit_predict(X_scaled)

    sil_score = silhouette_score(X_scaled, rfm_df["Cluster"])
    print(f"Silhouette Score: {sil_score:.4f}")

    metrics = {
        "n_clusters": n_clusters,
        "silhouette_score": round(sil_score, 4),
    }
    return rfm_df, metrics, scaler, kmeans
