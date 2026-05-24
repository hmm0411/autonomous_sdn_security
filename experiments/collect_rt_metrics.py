import time
import pandas as pd
from rl_engine.online_env import OnlineSDNEnv

SCENARIO = "portscanning"  # đổi tay khi test từng attack
DURATION = 60        # số giây log

def main():

    env = OnlineSDNEnv(
        controller_url="http://controller:8181/onos/v1"
    )

    state = env.reset()

    logs = []

    print(f"\n===== COLLECTING METRICS: {SCENARIO} =====\n")

    for step in range(DURATION):

        # Cập nhật chuẩn 8 chiều mới nhất
        packet_rate      = state[0]
        byte_rate        = state[1]
        flow_count       = state[2]
        flow_growth_rate = state[3]
        entropy          = state[4]
        latency          = state[5]
        drop_rate        = state[6]
        cpu              = state[7]

        print(
            f"[{step}] "
            f"Pkt={packet_rate:.2f} | "
            f"Byte={byte_rate:.2f} | "
            f"Flow={flow_count:.2f} | "
            f"Growth={flow_growth_rate:.2f} | "
            f"Drop={drop_rate:.4f} | "
            f"Lat={latency:.2f}"
        )

        logs.append({
            "time": step,
            "packet_rate": packet_rate,
            "byte_rate": byte_rate,
            "flow_count": flow_count,
            "flow_growth_rate": flow_growth_rate,
            "entropy": entropy,
            "latency": latency,
            "drop_rate": drop_rate,
            "controller_cpu": cpu,
            "scenario": SCENARIO
        })

        # Do môi trường online env trả về tuple 4 giá trị + info (hoặc 5 tùy version Gym)
        # Bắt đủ số lượng trả về tránh lỗi ValueError
        next_state, reward, terminated, truncated, info = env.step(0)
            
        state = next_state
        time.sleep(1)

    df = pd.DataFrame(logs)

    filename = f"results/metrics_{SCENARIO.lower()}.csv"
    df.to_csv(filename, index=False)

    print(f"\nSaved to {filename}")

if __name__ == "__main__":
    main()