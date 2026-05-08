# collector.py
import requests
import time
import numpy as np

class ONOSCollector:
    def __init__(self, onos_ip="34.126.64.185", port="8181", user="onos", pwd="rocks"):
        self.base_url = f"http://{onos_ip}:{port}/onos/v1"
        self.auth = (user, pwd)
        self.headers = {'Accept': 'application/json'}

    def get_state(self):
        """
        Gọi API ONOS để lấy thống kê mạng và trả về state vector 9 chiều.
        """
        try:
            # 1. Lấy thông tin Flows
            flows_response = requests.get(f"{self.base_url}/flows", auth=self.auth, headers=self.headers)
            flows_data = flows_response.json()
            
            # 2. Lấy thông tin Ports (để tính packet_rate, byte_rate, packet_loss)
            ports_response = requests.get(f"{self.base_url}/statistics/ports", auth=self.auth, headers=self.headers)
            ports_data = ports_response.json()

            # --- TRÍCH XUẤT VÀ TÍNH TOÁN FEATURE ---
            # Lưu ý: Đây là logic tính toán mẫu. Bạn cần điều chỉnh công thức 
            # cho khớp với cách bạn đã tính lúc tạo dataset trên Mininet/Kaggle.
            
            flow_count = len(flows_data.get('flows', []))
            
            # Khởi tạo các giá trị (Gắn giá trị mặc định nếu chưa tính được ngay)
            packet_rate = 0.0
            byte_rate = 0.0
            src_ip_entropy = 0.0 # Cần tính từ danh sách source IP của flows
            latency = 0.05       # ONOS không đo latency trực tiếp, cần script ping/OVS đo ngoài
            packet_loss = 0.0
            queue_length = 0.0
            controller_cpu = 0.1 # Có thể lấy từ thư viện psutil của máy chạy controller
            attack_indicator = 0.0 # Bằng 1 nếu vượt ngưỡng bất thường

            # Mảng 9 features khớp với file train_surrogate.py
            state = [
                packet_rate, 
                byte_rate, 
                flow_count, 
                src_ip_entropy, 
                latency, 
                packet_loss, 
                queue_length, 
                controller_cpu, 
                attack_indicator
            ]
            
            return state

        except Exception as e:
            print(f"[Collector Error] Không thể kết nối tới ONOS: {e}")
            return None