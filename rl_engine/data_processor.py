import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import joblib

def process_sdn_dataset():

    RAW_DATA_DIR = "data/raw"
    PROCESSED_DATA_DIR = "data/processed"   
    MODELS_DIR = "models"
    
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 1. Khai báo các file CSV
    files = {
        "data\\raw\\normal.csv": 0,
        "data\\raw\\ddos.csv": 1,
        "data\\raw\\flow_overflow.csv": 2,
        "data\\raw\\ip_spoofing.csv": 3,
        "data\\raw\\packet_in_flood.csv": 4,
        "data\\raw\\port_scanning.csv": 5
    }
    
    # 2. Đọc và gộp toàn bộ dữ liệu
    dfs = []
    for file, label in files.items():
        df = pd.read_csv(file)
        df['label'] = label
        dfs.append(df)
        print(f"Đã nạp {file} - Shape: {df.shape}")
        
    master_df = pd.concat(dfs, ignore_index=True)
    print(f"\nTổng số mẫu dữ liệu (Total Samples): {master_df.shape[0]}")

    # 3. Bổ sung các Feature bị khuyết (Nội suy giả lập)
    labels = master_df['label'].values
    
    entropy = np.zeros(len(master_df))
    queue = np.zeros(len(master_df))
    cpu = np.zeros(len(master_df))
    
    # Tạo dữ liệu giả lập có logic (Tấn công thì thông số cao hơn)
    for i, label in enumerate(labels):
        if label == 0: # Normal
            entropy[i] = np.random.uniform(0.1, 0.3)
            queue[i] = np.random.uniform(0.0, 0.2)
            cpu[i] = np.random.uniform(0.1, 0.3)
        elif label == 1: # DDoS: Băng thông cao -> Hàng đợi (queue) đầy, CPU cao
            entropy[i] = np.random.uniform(0.4, 0.6)
            queue[i] = np.random.uniform(0.8, 1.0)
            cpu[i] = np.random.uniform(0.7, 0.9)
            
        elif label == 2: # Packet_in: Bắn gói Packet-In liên tục -> CPU Controller quá tải
            entropy[i] = np.random.uniform(0.3, 0.5)
            queue[i] = np.random.uniform(0.5, 0.7)
            cpu[i] = np.random.uniform(0.9, 1.0)
            
        elif label == 3: # Flow_overflow: Bảng luồng của switch đầy -> Hàng đợi đầy
            entropy[i] = np.random.uniform(0.3, 0.5)
            queue[i] = np.random.uniform(0.9, 1.0)
            cpu[i] = np.random.uniform(0.5, 0.7)
            
        elif label == 4: # Spoofing: IP giả mạo ngẫu nhiên liên tục -> Entropy cực cao
            entropy[i] = np.random.uniform(0.8, 1.0)
            queue[i] = np.random.uniform(0.3, 0.5)
            cpu[i] = np.random.uniform(0.4, 0.6)
            
        elif label == 5: # Port_scan: Quét port -> Ít ảnh hưởng queue/cpu
            entropy[i] = np.random.uniform(0.2, 0.4)
            queue[i] = np.random.uniform(0.1, 0.3)
            cpu[i] = np.random.uniform(0.2, 0.4)
            
    master_df['src_ip_entropy'] = entropy
    master_df['queue_length'] = queue
    master_df['controller_cpu'] = cpu

    # Đổi tên cột drop_rate thành packet_loss
    master_df.rename(columns={'drop_rate': 'packet_loss'}, inplace=True)

    # 4. Sắp xếp lại thứ tự cột cho khớp với State
    feature_cols = [
        'packet_rate', 'byte_rate', 'flow_count', 'src_ip_entropy', 
        'latency', 'packet_loss', 'queue_length', 'controller_cpu'
    ]
    
    features = master_df[feature_cols]
    
    # 5. Chuẩn hóa dữ liệu (Min-Max Scaling) về khoảng [0, 1]
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Tạo DataFrame mới từ dữ liệu đã chuẩn hóa
    processed_df = pd.DataFrame(scaled_features, columns=feature_cols)
    
    # Thêm attack_indicator (normalized: 0-1)
    processed_df['attack_indicator'] = master_df['label'].values / 5.0
    
    # Thêm previous_action feature (khởi tạo = 0)
    processed_df['previous_action'] = 0.0
    
    # Split train/test trên dữ liệu đã chuẩn hóa
    train_frames = []
    test_frames = []

    for label in range(6):
        group_df = processed_df[processed_df['attack_indicator'] == label / 5.0].copy()
        group_df = group_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        split_idx = int(0.8 * len(group_df))
        train_frames.append(group_df.iloc[:split_idx])
        test_frames.append(group_df.iloc[split_idx:])
    
    train_df = pd.concat(train_frames, ignore_index=True)
    test_df = pd.concat(test_frames, ignore_index=True)
    
    print(f"\nKích thước tập Train: {train_df.shape}, Tập Test: {test_df.shape}")

    # 6. Lưu ra file CSV
    print("\nBước 6: Lưu dữ liệu đầu ra và Model Scaler...")
    train_path = os.path.join(PROCESSED_DATA_DIR, "train_data.csv")
    test_path = os.path.join(PROCESSED_DATA_DIR, "test_data.csv")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)  
    joblib.dump(scaler, scaler_path)

    print(f"Dữ liệu đã được lưu tại: {train_path} và {test_path}")
    print(f"Scaler đã được lưu tại: {scaler_path}")
    print(f"Các cột trong dữ liệu: {list(train_df.columns)}")

if __name__ == "__main__":
    process_sdn_dataset()