# experiments/test_online_rl.py
import pandas as pd

from rl_engine.online_env import OnlineSDNEnv
from rl_engine.agent.dqn_agent import DQNAgent
from rl_engine.agent.ppo_agent import PPOAgent
from rl_engine.config import STATE_DIM, ACTION_DIM

def main():
    env = OnlineSDNEnv(
        controller_url="http://controller:8181/onos/v1"
    )

    timeline_dqn = []
    timeline_ppo = []

    # ===== Load DQN =====
    dqn_agent = DQNAgent(STATE_DIM, ACTION_DIM)
    dqn_agent.load("models/dqn_model.pth")
    dqn_agent.epsilon = 0.0

    # ===== Load PPO =====
    ppo_agent = PPOAgent(STATE_DIM, ACTION_DIM)
    ppo_agent.load("models/ppo_model.pth")

    print("\n========== DQN ONLINE TEST ==========\n")

    state = env.reset()

    for step in range(50):
        action_tuple = dqn_agent.select_action(state)
        action = action_tuple[0] if isinstance(action_tuple, tuple) else action_tuple
        
        # CẬP NHẬT: Hứng đủ 5 giá trị theo chuẩn Gymnasium mới
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        timeline_dqn.append({
            "step": step,
            # CẬP NHẬT: Bỏ dấu [] và chỉnh lại index theo chuẩn 8 features
            "flow_count": next_state[2],
            "entropy": next_state[4],  
            "latency": next_state[5],  
            "action": action,
            "reward": reward
        })
        
        # Tối ưu logic in ra màn hình
        attack_status = env._detect_attack(state) or env._detect_attack(next_state)
        print(f"[DQN] Step {step} | Action: {action} | Reward: {reward:.2f} | Attack: {attack_status}")
        
        state = next_state

    df_dqn = pd.DataFrame(timeline_dqn)
    df_dqn.to_csv("results/dqn_online_test.csv", index=False)

    print("\n========== DQN ONLINE TEST COMPLETE ==========\n")


    print("\n========== PPO ONLINE TEST ==========\n")

    state = env.reset()

    for step in range(50):
        action_tuple = ppo_agent.select_action(state)
        action = action_tuple[0] if isinstance(action_tuple, tuple) else action_tuple
        
        # CẬP NHẬT: Hứng đủ 5 giá trị theo chuẩn Gymnasium mới
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        timeline_ppo.append({
            "step": step,
            # CẬP NHẬT: Bỏ dấu [] và chỉnh lại index theo chuẩn 8 features
            "flow_count": next_state[2],
            "entropy": next_state[4],  
            "latency": next_state[5],  
            "action": action,
            "reward": reward
        })

        attack_status = env._detect_attack(state) or env._detect_attack(next_state)
        print(f"[PPO] Step {step} | Action: {action} | Reward: {reward:.2f} | Attack: {attack_status}")
        
        state = next_state

    df_ppo = pd.DataFrame(timeline_ppo)
    df_ppo.to_csv("results/ppo_online_test.csv", index=False)

    print("\n========== PPO ONLINE TEST COMPLETE ==========\n")

if __name__ == "__main__":
    main()