import requests

# Khai báo cứng cả 2 địa chỉ
PPO_URL = "http://rl-agent-ppo:9001/predict"
DQN_URL = "http://rl-agent-dqn:9000/predict"

def get_action(state):
    try:
        state_list = [float(x) for x in state]
        payload = {"state": state_list}
        
        # BƯỚC 1: ƯU TIÊN PPO 
        try:
            response = requests.post(PPO_URL, json=payload, timeout=2)
            if response.status_code == 200:
                action = response.json().get("action", 0)
                return action
        except requests.exceptions.RequestException:
            pass 

        # BƯỚC 2: TỰ ĐỘNG CHUYỂN SANG DQN 
        try:
            response = requests.post(DQN_URL, json=payload, timeout=2)
            if response.status_code == 200:
                action = response.json().get("action", 0)
                return action
        except requests.exceptions.RequestException as e:
            print(f"Cả DQN và PPO đều mất kết nối: {e}")
            
        # Nếu cả 2 đều sập, trả về 0 để giữ mạng không bị cấu hình sai
        return 0
        
    except Exception as e:
        print(f"Lỗi logic khi chuẩn bị dữ liệu: {e}")
        return 0