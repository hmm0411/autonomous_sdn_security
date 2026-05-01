import requests

ONOS_URL = "http://controller:8181/onos/v1"
AUTH = ("onos", "rocks")

def execute_action(action):
    try:
        if action == 0:
            print("Quyết định: Bình thường (No Action)")
            pass
    #           no_action_id: int = 0
    # block_action_id: int = 1
    # limit_bw_action_id: int = 2
    # redirect_action_id: int = 3
    # isolate_action_id: int = 4
        elif action == 1:
            print("Quyết định: TẤN CÔNG! Ra lệnh chặn dòng dữ liệu...")
            # Ví dụ: Xóa tất cả các flow rule để chặn mạng tạm thời
            # Hoặc bạn có thể gọi API add flow rule để drop IP cụ thể
            requests.delete(f"{ONOS_URL}/flows/application/org.onosproject.cli", auth=AUTH)
            print("Đã áp dụng policy chặn mạng lên ONOS.")
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