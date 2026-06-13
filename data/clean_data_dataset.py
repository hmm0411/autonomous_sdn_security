import os
import glob
import numpy as np
import pandas as pd


RAW_DIR = "data/raw"
CLEAN_DIR = "data/processed"

os.makedirs(CLEAN_DIR, exist_ok=True)

FEATURES = [
    "packet_rate",
    "byte_rate",
    "flow_count",
    "flow_growth_rate",
    "src_ip_entropy",
    "latency",
    "packet_loss",
    "controller_cpu",
]

FILES = {
    "normal.csv": 0,
    "ddos.csv": 1,
    "flow_overflow.csv": 2,
    "ip_spoofing.csv": 3,
    "packet_in_flood.csv": 4,
    "port_scanning.csv": 5,
}


def clean_one_file(path, label):
    df = pd.read_csv(path)

    # 1. Đảm bảo đủ cột
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")

    # 2. Gán lại label theo file để tránh nhầm label
    df["label"] = label

    # 3. Xóa dòng NaN/inf
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURES)

    # 4. Bỏ 5-10 dòng đầu do warm-up
    if len(df) > 10:
        df = df.iloc[10:].reset_index(drop=True)

    # 5. Không giữ dòng lỗi quá rõ do collector fail
    # Nhưng không xóa quá mạnh để còn giống realtime.
    # Chỉ xóa dòng vừa latency=500 vừa flow_count bất thường hoặc CPU quá cao.
    bad = (
        ((df["latency"] >= 500) & (df["packet_loss"] >= 1.0)) |
        (df["controller_cpu"] > 250) |
        (df["flow_count"] < 0)
    )
    df = df[~bad].reset_index(drop=True)

    # 6. Clip outlier theo ngưỡng hợp lý cho Mininet/ONOS lab
    df["packet_rate"] = df["packet_rate"].clip(lower=0)
    df["byte_rate"] = df["byte_rate"].clip(lower=0)
    df["flow_count"] = df["flow_count"].clip(lower=0, upper=500)
    df["flow_growth_rate"] = df["flow_growth_rate"].clip(lower=0, upper=50)
    df["src_ip_entropy"] = df["src_ip_entropy"].clip(lower=0, upper=8)
    df["latency"] = df["latency"].clip(lower=0, upper=100)
    df["packet_loss"] = df["packet_loss"].clip(lower=0, upper=0.5)
    df["controller_cpu"] = df["controller_cpu"].clip(lower=0, upper=100)

    # 7. Log-transform cho rate để giảm lệch scale
    df["packet_rate"] = np.log1p(df["packet_rate"])
    df["byte_rate"] = np.log1p(df["byte_rate"])

    # 8. Làm mượt rolling để tránh attack chỉ spike 1-2 dòng rồi mất
    # Rolling max cho event-like features
    df["flow_growth_rate"] = (
        df["flow_growth_rate"]
        .rolling(window=5, min_periods=1)
        .max()
    )

    # Rolling mean nhẹ cho rate/cpu
    for col in ["packet_rate", "byte_rate", "controller_cpu"]:
        df[col] = (
            df[col]
            .rolling(window=3, min_periods=1)
            .mean()
        )

    # 9. Giữ đúng thứ tự cột
    df = df[FEATURES + ["label"]]

    return df


def main():
    for filename, label in FILES.items():
        path = os.path.join(RAW_DIR, filename)

        if not os.path.exists(path):
            print(f"[WARN] Missing {path}, skip")
            continue

        df = clean_one_file(path, label)
        out = os.path.join(CLEAN_DIR, filename)
        df.to_csv(out, index=False)

        print(f"[OK] {filename}: {df.shape}")
        print(df.describe())


if __name__ == "__main__":
    main()