import datetime

import numpy as np
import pandas as pd
import torch
import os
from rl_engine.offline_env import OfflineSDNEnv
from rl_engine.agent.dqn_agent import DQNAgent  
from rl_engine.agent.ppo_agent import PPOAgent
from experiments.baseline_rule import BaselineRuleBasedAgent
from rl_engine.config import *

def evaluate_agent(env, agent, episodes=20, max_steps=1000):

    episode_rewards = []
    switching_rates = []

    for ep in range(episodes):
        state, _ = env.reset()
        if hasattr(agent, "reset"):
            agent.reset()  # Nếu agent có hàm reset, gọi nó để làm mới trạng thái nội bộ
        done = False

        total_reward = 0
        action_history = []
        steps = 0

        while not done and steps < max_steps:

            if hasattr(agent, "predict"):
                result = agent.predict(state)
                action = result[0] if isinstance(result, tuple) else result
            else:
                action = agent.select_action(state)

            action = int(action)

            next_state, reward, terminated, truncated, _ = env.step(action)

            total_reward += reward
            action_history.append(action)

            state = next_state
            done = terminated or truncated
            steps += 1

        switching_rate = np.mean(
            np.array(action_history[1:]) != np.array(action_history[:-1])
        )

        episode_rewards.append(total_reward)
        switching_rates.append(switching_rate)

    return {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_switching": np.mean(switching_rates)
    }

def main():
    # 1. Đường dẫn tương đối từ thư mục experiments/
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TEST_DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "test_data.csv")
    DQN_MODEL_PATH = os.path.join(ROOT_DIR, "models", "dqn_model.pth")
    PPO_MODEL_PATH = os.path.join(ROOT_DIR, "models", "ppo_model.pth")

    RESULTS_DIR = os.path.join(ROOT_DIR, "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 2. Khởi tạo môi trường trực tiếp bằng đường dẫn
    print(f"Khởi tạo môi trường với dữ liệu từ: {TEST_DATA_PATH}")
    df_test = pd.read_csv(TEST_DATA_PATH)
    env = OfflineSDNEnv(dataframe=df_test, max_steps_per_episode=1000)

    all_results = []

    # 3. Đánh giá DQN
    print("Evaluating DQN...")
    dqn = DQNAgent(STATE_DIM, ACTION_DIM)
    if os.path.exists(DQN_MODEL_PATH):
        checkpoint = torch.load(DQN_MODEL_PATH, map_location=torch.device('cpu'))
        dqn.q_net.load_state_dict(checkpoint["model_state_dict"])
        # dqn.q_net.load_state_dict(torch.load(DQN_MODEL_PATH, map_location=torch.device('cpu')))
        # dqn.q_net.load_state_dict(torch.load(DQN_MODEL_PATH))
        dqn.q_net.load_state_dict(checkpoint["model_state_dict"])
        dqn.q_net.eval()  # Đặt mô hình ở chế độ đánh giá
        
        res = evaluate_agent(env, dqn)
        res['model'] = "DQN"
        all_results.append(res)
    else:
        dqn_result = "Không tìm thấy model DQN."

    # 4. Đánh giá PPO
    print("Evaluating PPO...")
    # Nếu PPOAgent của bạn bọc thư viện stable_baselines3
    ppo = PPOAgent(STATE_DIM, ACTION_DIM)
    if os.path.exists(PPO_MODEL_PATH):
        ppo.load(PPO_MODEL_PATH)
        res = evaluate_agent(env, ppo)
        res['model'] = "PPO"
        all_results.append(res)
    else:
        ppo_result = "Không tìm thấy model PPO."

    # 5. Đánh giá Rule-based
    print("Evaluating Rule-based...")
    rule = BaselineRuleBasedAgent()
    res = evaluate_agent(env, rule)
    res['model'] = "Rule-based"
    all_results.append(res)

    #6 Lưu kết quả vào file CSV để tiện cho việc vẽ biểu đồ sau này
    results_df = pd.DataFrame(all_results)
    csv_filename = f"evaluation_summary.csv"
    results_csv_path = os.path.join(RESULTS_DIR, csv_filename)
    results_df.to_csv(results_csv_path, index=False)
    print(f"Đã lưu kết quả đánh giá vào: {results_csv_path}")


    # In kết quả tổng hợp
    print("\n" + "="*45)
    print(f"{'MODEL':<15} | {'REWARD':<10} | {'SWITCH RATE':<12}")
    print("-" * 45)
    for r in all_results:
        print(f"{r['model']:<15} | {r['mean_reward']:<10.2f} | {r['mean_switching']:<12.4f}")
    print("="*45)
    
if __name__ == "__main__":
    main()