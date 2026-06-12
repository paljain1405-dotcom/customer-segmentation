import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


def generate_sample_data(n_customers=500, n_transactions=5000, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    customer_ids = [f"C{str(i).zfill(4)}" for i in range(1, n_customers + 1)]
    products = [f"PROD{str(i).zfill(3)}" for i in range(1, 50)]
    start_date = datetime(2023, 1, 1)
    records = []
    for _ in range(n_transactions):
        customer = random.choice(customer_ids)
        product = random.choice(products)
        quantity = random.randint(1, 20)
        unit_price = round(random.uniform(0.5, 50.0), 2)
        invoice_date = start_date + timedelta(days=random.randint(0, 364))
        records.append({
            "InvoiceNo": f"INV{random.randint(100000, 999999)}",
            "StockCode": product,
            "Quantity": quantity,
            "InvoiceDate": invoice_date.strftime("%Y-%m-%d"),
            "UnitPrice": unit_price,
            "CustomerID": customer,
        })
    df = pd.DataFrame(records)
    df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]
    return df
