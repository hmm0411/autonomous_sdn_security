import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


DATA_PATH = os.getenv("TRANSITION_LOG", "logs/transition_log.csv")
MODEL_PATH = os.getenv("SURROGATE_MODEL", "models/surrogate_model.pkl")
OUT_DIR = os.getenv("TWIN_RESULT_DIR", "results/digital_twin")


FEATURE_COLS = [
    "packet_rate",
    "byte_rate",
    "flow_count",
    "flow_growth_rate",
    "src_ip_entropy",
    "latency",
    "packet_loss",
    "controller_cpu",
    "action",
]

TARGET_COLS = [
    "next_latency",
    "next_packet_loss",
]


def main():
    os.makedirs(
        os.path.dirname(MODEL_PATH) if os.path.dirname(MODEL_PATH) else ".",
        exist_ok=True,
    )

    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Transition data not found: {DATA_PATH}. "
            f"Run control loop with MODE=collect first."
        )

    df = pd.read_csv(DATA_PATH).dropna()

    missing_cols = [
        col
        for col in FEATURE_COLS + TARGET_COLS
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(f"Missing columns in transition log: {missing_cols}")

    if len(df) < 50:
        raise RuntimeError(
            f"Need at least 50 transition rows, got {len(df)}. "
            f"Collect more runtime data first."
        )

    X = df[FEATURE_COLS].astype(float).values
    y = df[TARGET_COLS].astype(float).values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=18,
        min_samples_split=4,
        n_jobs=-1,
        random_state=42,
    )

    print("[TWIN_TRAIN] Training surrogate model...", flush=True)

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "r2": float(r2_score(y_test, pred)),
        "rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }

    metrics_path = os.path.join(OUT_DIR, "surrogate_metrics.csv")
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    plt.figure(figsize=(7, 5))
    plt.scatter(y_test[:, 0], pred[:, 0], alpha=0.35)

    lo = min(float(y_test[:, 0].min()), float(pred[:, 0].min()))
    hi = max(float(y_test[:, 0].max()), float(pred[:, 0].max()))

    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("Real next latency")
    plt.ylabel("Predicted next latency")
    plt.title("Digital Twin Surrogate: Latency Prediction")
    plt.tight_layout()

    fig_path = os.path.join(OUT_DIR, "pred_vs_real_latency.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

    joblib.dump(model, MODEL_PATH)

    print("[TWIN_TRAIN] metrics:", metrics, flush=True)
    print(f"[TWIN_TRAIN] saved model: {MODEL_PATH}", flush=True)
    print(f"[TWIN_TRAIN] saved metrics: {metrics_path}", flush=True)
    print(f"[TWIN_TRAIN] saved figure: {fig_path}", flush=True)


if __name__ == "__main__":
    main()