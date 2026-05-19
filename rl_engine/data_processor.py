import os
import joblib
import pandas as pd
from sklearn.preprocessing import QuantileTransformer


def process_sdn_dataset():
    processed_dir = "data/processed"
    models_dir = "models"

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    files = {
        "data/processed/normal.csv": 0,
        "data/processed/ddos.csv": 1,
        "data/processed/flow_overflow.csv": 2,
        "data/processed/ip_spoofing.csv": 3,
        "data/processed/packet_in_flood.csv": 4,
        "data/processed/port_scanning.csv": 5
    }

    feature_cols = [
        "packet_rate",
        "byte_rate",
        "flow_count",
        "flow_growth_rate",
        "src_ip_entropy",
        "latency",
        "packet_loss",
        "controller_cpu"
    ]

    dfs = []

    for path, label in files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

        df = pd.read_csv(path)

        for col in feature_cols:
            if col not in df.columns:
                raise ValueError(f"{path} missing required column: {col}")

        df["label"] = label
        df = df[feature_cols + ["label"]].copy()
        df = df.replace([float("inf"), -float("inf")], 0).fillna(0)

        dfs.append(df)
        print(f"Loaded {path}: {df.shape}")

    master_df = pd.concat(dfs, ignore_index=True)
    print("Total samples:", master_df.shape)

    # Fit scaler trên toàn bộ dữ liệu đã làm sạch để giữ nhất quán với realtime.
    # Nếu muốn nghiêm ngặt nghiên cứu hơn, fit scaler chỉ trên train.
    scaler = QuantileTransformer(
        output_distribution="uniform",
        n_quantiles=min(1000, len(master_df)),
        random_state=42
    )

    scaled = scaler.fit_transform(master_df[feature_cols])

    processed_df = pd.DataFrame(scaled, columns=feature_cols)
    processed_df["attack_indicator"] = master_df["label"].values / 5.0
    processed_df["previous_action"] = 0.0

    train_frames = []
    val_frames = []
    test_frames = []

    for label in range(6):
        group = processed_df[
            processed_df["attack_indicator"] == label / 5.0
        ].copy().reset_index(drop=True)

        n = len(group)
        train_end = int(0.70 * n)
        val_end = int(0.85 * n)

        train_frames.append(group.iloc[:train_end])
        val_frames.append(group.iloc[train_end:val_end])
        test_frames.append(group.iloc[val_end:])

        print(
            f"Label {label}: total={n}, "
            f"train={train_end}, "
            f"val={val_end - train_end}, "
            f"test={n - val_end}"
        )

    train_df = pd.concat(train_frames, ignore_index=True)
    val_df = pd.concat(val_frames, ignore_index=True)
    test_df = pd.concat(test_frames, ignore_index=True)

    train_path = os.path.join(processed_dir, "train_data.csv")
    val_path = os.path.join(processed_dir, "val_data.csv")
    test_path = os.path.join(processed_dir, "test_data.csv")
    scaler_path = os.path.join(models_dir, "scaler.pkl")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    joblib.dump(scaler, scaler_path)

    print("Saved:", train_path)
    print("Saved:", val_path)
    print("Saved:", test_path)
    print("Saved:", scaler_path)

    print("Columns:", list(train_df.columns))
    print("Train shape:", train_df.shape)
    print("Val shape:", val_df.shape)
    print("Test shape:", test_df.shape)


if __name__ == "__main__":
    process_sdn_dataset()