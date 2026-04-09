# train_surrogate.py

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_PATH = "transition_dataset.csv"
MODEL_PATH = "surrogate_model.pkl"


FEATURE_COLS = [
    "packet_rate",
    "byte_rate",
    "flow_count",
    "src_ip_entropy",
    "latency",
    "packet_loss",
    "queue_length",
    "controller_cpu",
    "attack_indicator",
    "action"
]

TARGET_COLS = [
    "next_latency",
    "next_packet_loss"
]


def load_data():
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURE_COLS].values
    y = df[TARGET_COLS].values

    attack_labels = df["attack_type"].values

    return train_test_split(
        X, y, attack_labels,
        test_size=0.2,
        random_state=42
    )


def evaluate(model, X_test, y_test, attack_labels):

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("\n==== GLOBAL METRICS ====")
    print("MAE :", mae)
    print("RMSE:", rmse)
    print("R2  :", r2)

    # --- Per attack evaluation ---
    print("\n==== PER-ATTACK METRICS ====")

    unique_attacks = np.unique(attack_labels)

    for attack in unique_attacks:
        idx = attack_labels == attack
        if np.sum(idx) == 0:
            continue

        attack_mae = mean_absolute_error(y_test[idx], preds[idx])
        attack_rmse = np.sqrt(mean_squared_error(y_test[idx], preds[idx]))
        attack_r2 = r2_score(y_test[idx], preds[idx])

        print(f"\nAttack: {attack}")
        print("  MAE :", attack_mae)
        print("  RMSE:", attack_rmse)
        print("  R2  :", attack_r2)

    return preds


def plot_prediction_vs_real(y_test, preds):

    plt.figure(figsize=(10,5))

    plt.scatter(y_test[:,0], preds[:,0], alpha=0.3)
    plt.xlabel("Real Next Latency")
    plt.ylabel("Predicted Next Latency")
    plt.title("Latency Prediction Scatter")
    plt.plot([0,1],[0,1], 'r--')

    plt.show()


def train():

    X_train, X_test, y_train, y_test, attack_train, attack_test = load_data()

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=18,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )

    print("Training surrogate model...")
    model.fit(X_train, y_train)

    preds = evaluate(model, X_test, y_test, attack_test)

    plot_prediction_vs_real(y_test, preds)

    joblib.dump(model, MODEL_PATH)
    print("\nModel saved to:", MODEL_PATH)


if __name__ == "__main__":
    train()