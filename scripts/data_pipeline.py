"""
Shared data loading, cleaning, and evaluation helpers.

Mirrors the cleaning / feature-engineering cells in
comparison-of-classification-regression-rnn.ipynb (cells 6-17, 29, 91-99),
so every model script (Decision Tree, Random Forest, Neural Network, ...)
starts from an identical, reproducible dataset.
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

DATA_PATH = "DataCoSupplyChainDataset.csv"


def load_and_clean(path=DATA_PATH):
    dataset = pd.read_csv(path, header=0, encoding="unicode_escape")
    print(f"Loaded dataset: {dataset.shape[0]} rows, {dataset.shape[1]} columns")

    dataset["Customer Full Name"] = (
        dataset["Customer Fname"].astype(str) + dataset["Customer Lname"].astype(str)
    )

    data = dataset.drop(
        [
            "Customer Email",
            "Product Status",
            "Customer Password",
            "Customer Street",
            "Customer Fname",
            "Customer Lname",
            "Latitude",
            "Longitude",
            "Product Description",
            "Product Image",
            "Order Zipcode",
            "shipping date (DateOrders)",
        ],
        axis=1,
    )
    data["Customer Zipcode"] = data["Customer Zipcode"].fillna(0)

    # Date-derived features (notebook cell 29). `.weekday_name` was removed from
    # pandas after 0.23, so we use the modern `.day_name()` equivalent.
    order_dates = pd.DatetimeIndex(data["order date (DateOrders)"])
    data["order_year"] = order_dates.year
    data["order_month"] = order_dates.month
    data["order_week_day"] = order_dates.day_name()
    data["order_hour"] = order_dates.hour
    data["order_month_year"] = pd.to_datetime(data["order date (DateOrders)"]).dt.to_period("M")

    return data


def engineer_features(data):
    train_data = data.copy()
    train_data["fraud"] = np.where(train_data["Order Status"] == "SUSPECTED_FRAUD", 1, 0)
    train_data["late_delivery"] = np.where(
        train_data["Delivery Status"] == "Late delivery", 1, 0
    )
    train_data.drop(
        [
            "Delivery Status",
            "Late_delivery_risk",
            "Order Status",
            "order_month_year",
            "order date (DateOrders)",
        ],
        axis=1,
        inplace=True,
    )

    categorical_cols = [
        "Customer Country",
        "Market",
        "Type",
        "Product Name",
        "Customer Segment",
        "Customer State",
        "Order Region",
        "Order City",
        "Category Name",
        "Customer City",
        "Department Name",
        "Order State",
        "Shipping Mode",
        "order_week_day",
        "Order Country",
        "Customer Full Name",
    ]
    le = preprocessing.LabelEncoder()
    for col in categorical_cols:
        train_data[col] = le.fit_transform(train_data[col])

    return train_data


def load_training_data(path=DATA_PATH):
    """Convenience wrapper: returns the fully cleaned + encoded train_data frame."""
    return engineer_features(load_and_clean(path))


def evaluate(name, y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf = confusion_matrix(y_test, y_pred)

    print(f"\n--- {name} ---")
    print(f"Accuracy  : {accuracy * 100:.2f}%")
    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall * 100:.2f}%")
    print(f"F1 score  : {f1 * 100:.2f}%")
    print(f"Confusion matrix:\n{conf}")

    return {
        "task": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
