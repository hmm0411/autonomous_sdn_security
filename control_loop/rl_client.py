import requests

DQN_URL = "http://rl-agent-dqn:9000/predict"
PPO_URL = "http://rl-agent-ppo:9001/predict"

def call_model(url, state):
    try:
        res = requests.post(url, json={"state": state.tolist()}, timeout=1.5)
        return int(res.json()["action"])
    except:
        return 0

def get_best_action(state):
    action_dqn = call_model(DQN_URL, state)
    action_ppo = call_model(PPO_URL, state)

    print("DQN action:", action_dqn)
    print("PPO action:", action_ppo)

    return action_ppo, "PPO"
