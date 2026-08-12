"""
Advanced Model Tier: MLP Neural Network Classification
Starter prepared by Kenneth Kamau; reviewed and finalized by Tyriq Angili.

Tasks: (1) Fraud detection  (2) Late delivery prediction
Compares two learning rates: 0.001 (default) vs 0.01.

Usage:
    python scripts/neural_network_mlp.py
Expects the dataset at the repo root: DataCoSupplyChainDataset.csv
"""

import time
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from data_pipeline import load_training_data

TASKS = ["fraud", "late_delivery"]

# Good idea to test two learning rates - this will help us see which one gives better results
LEARNING_RATES = [0.001, 0.01]


def split_and_scale(train_data, task):
    x = train_data.loc[:, train_data.columns != task]
    y = train_data[task]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test, y_train, y_test


def main():
    train_data = load_training_data()
    results = []

    for task in TASKS:
        x_train, x_test, y_train, y_test = split_and_scale(train_data, task)

        for lr in LEARNING_RATES:
            model = MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=30,
                random_state=0,
                learning_rate_init=lr,
            )

            t0 = time.time()
            model.fit(x_train, y_train)
            train_time = time.time() - t0

            y_pred = model.predict(x_test)

            row = {
                "task": task,
                "learning_rate": lr,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "train_time_sec": round(train_time, 2),
            }
            results.append(row)
            print(
                f"[{task:14s}] lr={lr:<6} "
                f"acc={row['accuracy']*100:5.2f}%  prec={row['precision']*100:5.2f}%  "
                f"rec={row['recall']*100:5.2f}%  f1={row['f1']*100:5.2f}%  "
                f"train_time={row['train_time_sec']}s"
            )

    summary = pd.DataFrame(results)
    print("\n=== Summary ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
# Reviewed and Finalized by Tyriq Angili
