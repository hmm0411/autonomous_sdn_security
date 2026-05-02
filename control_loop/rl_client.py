import requests

DQN_URL = "http://rl-agent-dqn:9000/predict"
PPO_URL = "http://rl-agent-ppo:9001/predict"

print(">>> HYBRID VERSION LOADED <<<")

def call_model(url, state):
    try:
        res = requests.post(url, json={"state": state.tolist()}, timeout=1.5)
        return int(res.json()["action"])
    except:
        return 0

def get_best_action(state):
    """
    Hybrid Arbitration:
    - Attack regime (pressure cao)  → ưu tiên DQN
    - Normal regime                → ưu tiên PPO
    """

    action_dqn = call_model(DQN_URL, state)
    action_ppo = call_model(PPO_URL, state)

    # state layout (scaled):
    # [packet_rate, byte_rate, flow_ratio, entropy, latency,
    #  packet_loss, queue_ratio, cpu, previous_action]
    flow_ratio = state[2]
    entropy = state[3]
    queue_ratio = state[6]
    cpu = state[7]

    pressure = 0.4 * entropy + 0.3 * flow_ratio + 0.2 * queue_ratio + 0.1 * cpu

    print("---- DECISION DEBUG ----")
    print("Entropy:", entropy)
    print("Flow ratio:", flow_ratio)
    print("Queue ratio:", queue_ratio)
    print("CPU:", cpu)
    print("PRESSURE:", pressure)
    print("DQN action:", action_dqn)
    print("PPO action:", action_ppo)

    # Attack regime
    if pressure > 0.5:
        if action_dqn != 0:
            print(">> ATTACK MODE: Use DQN")
            return action_dqn, "DQN-ATTACK", 0
        else:
            print(">> ATTACK MODE but DQN=0 → fallback PPO")
            return action_ppo, "PPO-FALLBACK", 0

    # Normal regime
    print(">> NORMAL MODE: Use PPO")
    return action_ppo, "PPO-NORMAL", 0
