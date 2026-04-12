import requests
# 'rl-agent' là tên service trong docker-compose, port 8000 là port API của bạn
RL_API_URL = "http://rl-agent:8000/predict"

def get_action(state):
    try:
        payload = {"state": state}
        response = requests.post(RL_API_URL, json=payload)
        response.raise_for_status()
        
        # Giả định API trả về JSON: {"action": 1}
        action = response.json().get("action", 0)
        return action
    except Exception as e:
        print(f"Lỗi khi gọi RL API: {e}")
        return 0 # Trả về action mặc định (ví dụ: 0 = Do nothing)