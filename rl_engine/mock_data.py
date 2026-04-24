import pandas as pd
import numpy as np
import os

# Tạo sẵn thư mục lưu data
os.makedirs('data/processed', exist_ok=True)

# Giả lập 500 dòng dữ liệu Offline với các thông số ngẫu nhiên từ 0-1
np.random.seed(42)
dummy_data = {
    'packet_rate': np.random.rand(500),
    'byte_rate': np.random.rand(500),
    'flow_count': np.random.rand(500),
    'src_ip_entropy': np.random.rand(500),
    'latency': np.random.rand(500),
    'packet_loss': np.random.rand(500),
    'queue_length': np.random.rand(500),
    'controller_cpu': np.random.rand(500),
    'attack_indicator': np.random.randint(0, 6, 500) / 5.0, # Nhãn các loại tấn công mô phỏng
    'previous_action': np.zeros(500)
}

df = pd.DataFrame(dummy_data)
df.to_csv('data/processed/test_data.csv', index=False)
print("Đã tạo file data/processed/test_data.csv giả lập thành công!")
