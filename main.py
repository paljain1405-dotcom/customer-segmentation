import pandas as pd
import os
from src.data_generator import generate_sample_data
from src.rfm_analysis import run_rfm_analysis
from src.clustering import run_kmeans_clustering

os.makedirs("data", exist_ok=True)

print("Step 1: Generating data...")
df = generate_sample_data()
df.to_csv("data/retail_transactions.csv", index=False)
print(f"Generated {len(df)} transactions")

print("Step 2: RFM Analysis...")
rfm_df = run_rfm_analysis(df)
rfm_df.to_csv("data/rfm_scores.csv", index=False)

print("Step 3: Clustering...")
clustered_df, metrics, _, _ = run_kmeans_clustering(rfm_df)
clustered_df.to_csv("data/clustered_customers.csv", index=False)

print("Done! Silhouette Score:", metrics["silhouette_score"])
