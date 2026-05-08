import time
import pandas as pd
from rl_engine.online_env import OnlineSDNEnv


class DemoDefenseAgent:

    def __init__(self, env):
        self.env = env
        self.baseline_packet = None
        self.baseline_flow = None
        self.baseline_drop = None
        self.warmup_steps = 10  # 10 bước đầu chỉ học baseline

    # ==========================================
    # BASELINE UPDATE (chỉ update khi NORMAL)
    # ==========================================
    def _update_baseline(self, packet_rate, flow_count, drop_rate):
        # If any are None, initialize them immediately and stop
        if self.baseline_packet is None or self.baseline_flow is None or self.baseline_drop is None:
            self.baseline_packet = packet_rate
            self.baseline_flow = flow_count
            self.baseline_drop = drop_rate
            return # Exit the function so the math below is safe

        # Pylance now knows these are not None
        alpha = 0.1
        self.baseline_packet = (1 - alpha) * self.baseline_packet + alpha * packet_rate
        self.baseline_flow = (1 - alpha) * self.baseline_flow + alpha * flow_count
        self.baseline_drop = (1 - alpha) * self.baseline_drop + alpha * drop_rate

    # ==========================================
    # DETECTION
    # ==========================================
    def detect_attack(self, packet_rate, flow_count, drop_rate):
        # 1. Guard Clause: If baseline isn't set, we can't calculate deltas.
        # This fixes the Pylance "None" operator error.
        if self.baseline_packet is None or self.baseline_flow is None or self.baseline_drop is None:
            return "NORMAL"

        # Now Pylance knows these are floats, not None
        delta_packet = packet_rate - self.baseline_packet
        delta_flow = flow_count - self.baseline_flow
        delta_drop = drop_rate - self.baseline_drop

        # ===== DDOS =====
        if delta_packet > self.baseline_packet * 3:
            return "DDOS"

        # ===== FLOW OVERFLOW =====
        if delta_flow > self.baseline_flow * 1.5:  # Changed to 1.5 based on your image
            return "FLOW_OVERFLOW"

        # ===== IP SPOOF =====
        if delta_drop > 0.05:  # Changed to 0.05 based on your image
            return "IP_SPOOF"

        # ===== PORT SCAN =====
        if delta_flow > self.baseline_flow * 1.2 and delta_packet < self.baseline_packet * 2:
            return "PORT_SCAN"

        return "NORMAL"

    # ==========================================
    # MAP ATTACK → ACTION
    # ==========================================
    def map_action(self, attack):

        if attack == "DDOS":
            return 2  # LIMIT

        if attack == "FLOW_OVERFLOW":
            return 4  # ISOLATE

        if attack == "IP_SPOOF":
            return 1  # BLOCK

        if attack == "PORT_SCAN":
            return 3  # REDIRECT

        return 0

    # ==========================================
    # MAIN STEP
    # ==========================================
    def step(self, state, step_count):

        packet_rate = state[0]
        flow_count = state[2]
        drop_rate = state[5]

        # Warmup baseline
        if step_count < self.warmup_steps:
            self._update_baseline(packet_rate, flow_count, drop_rate)
            return "BASELINE_LEARNING", 0

        attack = self.detect_attack(packet_rate, flow_count, drop_rate)

        # Chỉ update baseline nếu NORMAL
        if attack == "NORMAL":
            self._update_baseline(packet_rate, flow_count, drop_rate)

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

        attack, action = agent.step(state, step)

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


if __name__ == "__main__":
    main()