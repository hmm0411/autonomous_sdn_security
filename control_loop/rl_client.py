import requests

RL_URL = "http://rl-serving:8000/predict"

def get_action(state):
    try:
        res = requests.post(
            RL_URL,
            json={"state": state.tolist()},
            timeout=1.5
        )
        data = res.json()
        return int(data.get("action", 0)), data.get("model", "unknown")
    except Exception as e:
        print("RL API error:", e)
        return 0, "fallback"