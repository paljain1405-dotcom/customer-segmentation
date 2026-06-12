import pandas as pd
import numpy as np
from datetime import datetime


def compute_rfm(df, snapshot_date=None):
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    if snapshot_date is None:
        snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalAmount", "sum"),
    ).reset_index()
    rfm["Monetary"] = rfm["Monetary"].round(2)
    return rfm


def score_rfm(rfm):
    rfm = rfm.copy()
    rfm["R_Score"] = pd.qcut(rfm["Recency"], q=4, labels=[4, 3, 2, 1]).astype(int)
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), q=4, labels=[1, 2, 3, 4]).astype(int)
    rfm["M_Score"] = pd.qcut(rfm["Monetary"].rank(method="first"), q=4, labels=[1, 2, 3, 4]).astype(int)
    rfm["RFM_Score"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]
    return rfm


def run_rfm_analysis(df):
    rfm = compute_rfm(df)
    rfm = score_rfm(rfm)
    print("RFM Analysis complete!")
    print(rfm[["Recency", "Frequency", "Monetary"]].describe().round(2))
    return rfm
