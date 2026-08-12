"""
Part A - Individual Practical Assessment
Model: Decision Tree Classification (Basic tier)
Tasks: (1) Fraud detection  (2) Late delivery prediction

Usage:
    python scripts/decision_tree_model.py
Expects the dataset at: data/DataCoSupplyChainDataset.csv
"""

import time
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_pipeline import load_training_data, evaluate


def main():
    overall_start = time.time()
    train_data = load_training_data()

    results = []

    # --- Fraud detection ---
    xf = train_data.loc[:, train_data.columns != "fraud"]
    yf = train_data["fraud"]
    xf_train, xf_test, yf_train, yf_test = train_test_split(
        xf, yf, test_size=0.2, random_state=42
    )
    sc_f = StandardScaler()
    xf_train = sc_f.fit_transform(xf_train)
    xf_test = sc_f.transform(xf_test)

    model_f = tree.DecisionTreeClassifier(random_state=42)
    t0 = time.time()
    model_f.fit(xf_train, yf_train)
    fraud_train_time = time.time() - t0
    yf_pred = model_f.predict(xf_test)
    print(f"\nFraud detection model training time: {fraud_train_time:.2f}s")
    results.append(evaluate("Fraud detection", yf_test, yf_pred))

    # --- Late delivery prediction ---
    xl = train_data.loc[:, train_data.columns != "late_delivery"]
    yl = train_data["late_delivery"]
    xl_train, xl_test, yl_train, yl_test = train_test_split(
        xl, yl, test_size=0.2, random_state=42
    )
    sc_l = StandardScaler()
    xl_train = sc_l.fit_transform(xl_train)
    xl_test = sc_l.transform(xl_test)

    model_l = tree.DecisionTreeClassifier(random_state=42)
    t0 = time.time()
    model_l.fit(xl_train, yl_train)
    late_train_time = time.time() - t0
    yl_pred = model_l.predict(xl_test)
    print(f"\nLate delivery model training time: {late_train_time:.2f}s")
    results.append(evaluate("Late delivery prediction", yl_test, yl_pred))

    print(f"\nTotal runtime: {time.time() - overall_start:.2f}s")

    summary = pd.DataFrame(results)
    print("\n=== Summary ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
