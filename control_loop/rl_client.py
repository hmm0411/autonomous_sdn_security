import requests

DQN_URL = "http://rl-agent-dqn:9000/predict"
PPO_URL = "http://rl-agent-ppo:9001/predict"

def call_model(url, state):
    try:
        res = requests.post(url, json={"state": state.tolist()}, timeout=1.5)
        return int(res.json()["action"])
    except:
        return 0

def get_best_action(state, reward_fn):
    action_dqn = call_model(DQN_URL, state)
    action_ppo = call_model(PPO_URL, state)

    reward_dqn = reward_fn(state, action_dqn)
    reward_ppo = reward_fn(state, action_ppo)

    if reward_dqn >= reward_ppo:
        return action_dqn, "DQN", reward_dqn
    else:
        return action_ppo, "PPO", reward_ppo
