import time
import pandas as pd
from rl_engine.online_env import OnlineSDNEnv

class DemoDefenseAgent:

    def __init__(self, env):
        self.env = env
        self.previous_attack = "NORMAL"

    def detect_attack(self, packet_rate, flow_count, drop_rate):

        if packet_rate > 80000:
            return "DDOS"

        if flow_count > 120:
            return "FLOW_OVERFLOW"

        if drop_rate > 0.15:
            return "IP_SPOOF"

        if 20000 < packet_rate < 80000 and flow_count > 60:
            return "PORT_SCAN"

        return "NORMAL"

    def map_action(self, attack):

        if attack == "DDOS":
            return 2  # LIMIT

        if attack == "FLOW_OVERFLOW":
            return 4  # ISOLATE

        if attack == "IP_SPOOF":
            return 1  # BLOCK

        if attack == "PORT_SCAN":
            return 3  # REDIRECT

        return 0  # NORMAL

    def step(self, state):

        packet_rate = state[0]
        flow_count = state[2]
        drop_rate = state[5]

        attack = self.detect_attack(packet_rate, flow_count, drop_rate)
        action = self.map_action(attack)

        return attack, action


def main():

    env = OnlineSDNEnv(
        controller_url="http://34.126.64.185:8181/onos/v1"
    )

    agent = DemoDefenseAgent(env)

    state = env.reset()
    timeline = []

    print("\n===== DEMO DEFENSE START =====\n")

    for step in range(100):

        attack, action = agent.step(state)

        next_state, reward, _, _ = env.step(action)

        print(f"[Step {step}] Attack={attack} | Action={action} | Reward={reward}")

        timeline.append({
            "step": step,
            "packet_rate": next_state[0],
            "flow_count": next_state[2],
            "drop_rate": next_state[5],
            "attack": attack,
            "action": action,
            "reward": reward
        })

        state = next_state
        time.sleep(1)

    df = pd.DataFrame(timeline)
    df.to_csv("results/demo_timeline.csv", index=False)

    print("\n===== DEMO COMPLETE =====")
    print("Timeline saved to results/demo_timeline.csv")


if __name__ == "__main__":
    main()