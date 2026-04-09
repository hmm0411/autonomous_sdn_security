# controller_client.py
import requests
import json

class ControllerClient:
    def __init__(self, onos_ip="34.126.64.185", port="8181", user="onos", pwd="rocks"):
        self.base_url = f"http://{onos_ip}:{port}/onos/v1"
        self.auth = (user, pwd)
        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        # Cần xác định ID của switch mà bạn muốn áp dụng luật (vd: s1)
        self.target_device = "of:0000000000000001" 

    def apply_action(self, action_idx):
        """
        Dịch action index thành cấu hình mạng và đẩy xuống ONOS
        """
        # Giả sử mô hình của bạn có các action sau:
        # 0: Bình thường (Không làm gì)
        # 1: Block IP tấn công (Thả gói / Drop)
        # 2: Giới hạn băng thông (Rate limiting)
        
        if action_idx == 0:
            print("[Action] Hệ thống an toàn, không thay đổi flow.")
            return True
            
        elif action_idx == 1:
            print("[Action] Phát hiện tấn công! Gửi lệnh DROP xuống switch...")
            return self._send_drop_flow()
            
        # Thêm các action khác tùy thuộc vào RL model của bạn...
        
        return False

    def _send_drop_flow(self):
        """
        Gửi một flow rule xuống ONOS với instruction là NOACTION (DROP)
        """
        flow_rule = {
            "priority": 50000,
            "timeout": 60, # Tự động xóa sau 60s
            "isPermanent": False,
            "deviceId": self.target_device,
            "treatment": {
                "instructions": [
                    {
                        "type": "NOACTION" # Tương đương với Drop
                    }
                ]
            },
            "selector": {
                "criteria": [
                    {
                        "type": "ETH_TYPE",
                        "ethType": "0x0800" # IPv4
                    }
                    # Bạn có thể lấy IP nguồn từ state để block chính xác mục tiêu
                ]
            }
        }

        try:
            url = f"{self.base_url}/flows/{self.target_device}"
            response = requests.post(
                url, 
                auth=self.auth, 
                headers=self.headers, 
                data=json.dumps(flow_rule)
            )
            
            if response.status_code in [200, 201]:
                print("[ONOS] Đã cài đặt Flow Rule thành công.")
                return True
            else:
                print(f"[ONOS Error] Lỗi đẩy Flow: {response.text}")
                return False
                
        except Exception as e:
            print(f"[Controller Error] Không thể kết nối tới ONOS: {e}")
            return False