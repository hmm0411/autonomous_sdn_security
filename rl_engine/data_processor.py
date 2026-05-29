import os
import joblib
import pandas as pd
from sklearn.preprocessing import QuantileTransformer


def balance_class(df, target_size=3000, random_state=42):
    if len(df) > target_size:
        return df.sample(
            n=target_size,
            random_state=random_state
        ).reset_index(drop=True)

    if len(df) < target_size:
        extra = df.sample(
            n=target_size - len(df),
            replace=True,
            random_state=random_state
        )

        return pd.concat([df, extra], ignore_index=True).reset_index(drop=True)

    return df.reset_index(drop=True)


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

    target_size = 3000
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

        # Cân bằng class ngay tại đây, trước khi append
        df = balance_class(
            df,
            target_size=target_size,
            random_state=42 + label
        )

        dfs.append(df)

        print(f"Loaded and balanced {path}: {df.shape}, label={label}")

    master_df = pd.concat(dfs, ignore_index=True)

    print("\nTotal samples after balancing:", master_df.shape)
    print("Label distribution:")
    print(master_df["label"].value_counts().sort_index())

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
        attack_value = label / 5.0

        group = processed_df[
            processed_df["attack_indicator"] == attack_value
        ].copy().reset_index(drop=True)

        n = len(group)
        train_end = int(0.70 * n)
        val_end = int(0.85 * n)

        train_part = group.iloc[:train_end]
        val_part = group.iloc[train_end:val_end]
        test_part = group.iloc[val_end:]

        train_frames.append(train_part)
        val_frames.append(val_part)
        test_frames.append(test_part)

        print(
            f"Label {label}: total={n}, "
            f"train={len(train_part)}, "
            f"val={len(val_part)}, "
            f"test={len(test_part)}"
        )

    train_df = pd.concat(train_frames, ignore_index=True)
    val_df = pd.concat(val_frames, ignore_index=True)
    test_df = pd.concat(test_frames, ignore_index=True)

    train_path = os.path.join(processed_dir, "train_data.csv")
    val_path = os.path.join(processed_dir, "val_data.csv")
    test_path = os.path.join(processed_dir, "test_data.csv")
    scaler_path = os.path.join(models_dir, "scaler.pkl")

    train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    val_df = val_df.sample(frac=1.0, random_state=43).reset_index(drop=True)
    test_df = test_df.sample(frac=1.0, random_state=44).reset_index(drop=True)

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    joblib.dump(scaler, scaler_path)

    print("\nSaved:", train_path)
    print("Saved:", val_path)
    print("Saved:", test_path)
    print("Saved:", scaler_path)

    print("\nColumns:", list(train_df.columns))
    print("Train shape:", train_df.shape)
    print("Val shape:", val_df.shape)
    print("Test shape:", test_df.shape)

    print("\nTrain label distribution:")
    print(train_df["attack_indicator"].value_counts().sort_index())

    print("\nVal label distribution:")
    print(val_df["attack_indicator"].value_counts().sort_index())

    print("\nTest label distribution:")
    print(test_df["attack_indicator"].value_counts().sort_index())


if __name__ == "__main__":
    process_sdn_dataset()