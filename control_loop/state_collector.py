import requests
import numpy as np

# Sử dụng tên service 'onos' trong docker-compose thay vì localhost
ONOS_URL = "http://onos:8181/onos/v1"
AUTH = ("onos", "rocks") # User/Pass mặc định của ONOS

def get_state():
    try:
        # Ví dụ: Lấy danh sách flows
        response = requests.get(f"{ONOS_URL}/flows", auth=AUTH)
        response.raise_for_status()
        flows = response.json().get('flows', [])
        flow_count = len(flows)
        
        # TODO: CẬP NHẬT VECTOR NÀY THEO ĐÚNG STATE DIMENSION CỦA BẠN
        # Giả sử mô hình của bạn cần vector 5 chiều: [flow_count, 0, 0, 0, 0]
        state = [flow_count, 0.0, 0.0, 0.0, 0.0] 
        
        return state
    except Exception as e:
        print(f"Lỗi khi lấy state từ ONOS: {e}")
        # Trả về mảng 0 nếu lỗi để hệ thống không bị crash
        return [0.0, 0.0, 0.0, 0.0, 0.0]