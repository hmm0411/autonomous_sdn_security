import time

import requests

ONOS_URL = "http://controller:8181/onos/v1"
AUTH = ("onos", "rocks")

DEVICE_ID = "of:0000000000000001" 
ONOS_URL = "http://controller:8181/onos/v1"

def execute_action(action):
    headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
    # Không để return ở đây
    try:
        if action == 1:
            url = f"{ONOS_URL}/flows/{DEVICE_ID}"
            response = requests.delete(url, auth=AUTH, headers=headers)
            print(f"Action 1 status: {response.status_code}")
        # ... các action khác ...
    except Exception as e:
        print(f"Lỗi: {e}")
    try:
        if action == 1: # BLOCK
            # URL chuẩn của ONOS để xóa flow rule:
            url = f"{ONOS_URL}/flows/{DEVICE_ID}"
            response = requests.delete(url, auth=AUTH, headers=headers)
            print(f"Action 1 status: {response.status_code}")
        elif action == 2:
            print("Quyết định: Giới hạn băng thông (Rate Limit)")
            # Ví dụ: Gọi API để thêm flow rule với instruction là RATE_LIMIT
            # Cần xây dựng payload phù hợp với API của ONOS để áp dụng rate limit
        elif action == 3:
            print("Quyết định: Chuyển hướng (Redirect)")
            # Ví dụ: Gọi API để thêm một flow rule mới vào ONOS
            # Cần xây dựng payload phù hợp với API của ONOS để thêm flow rule
        elif action == 4:
            print("Quyết định: Cách ly (Isolate)")
            # Ví dụ: Gọi API để thêm một flow rule mới vào ONOS
            # Cần xây dựng payload phù hợp với API của ONOS để thêm flow rule

        else:
            print(f"Action không xác định: {action}")
            
    except Exception as e:
        print(f"Lỗi khi áp dụng action lên ONOS: {e}")