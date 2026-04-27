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

def evaluate_agent(env, agent, max_steps=1000):
    state, _ = env.reset()
    done = False
    
    total_reward = 0
    action_history = []
    rewards = []
    steps = 0

    while not done and steps < max_steps:
        # Xử lý cách lấy action tùy theo loại Agent
        if hasattr(agent, "predict"):
            try:
                action, _ = agent.predict(state, deterministic=True)
            except TypeError:
                result = agent.predict(state)
                action = result[0] if isinstance(result, tuple) else result
        elif hasattr(agent, "select_action"):
            action = agent.select_action(state)
        else:
            raise ValueError("Agent không có phương thức predict hoặc select_action.")

        # Chuyển kiểu action về int nếu cần thiết (để tránh lỗi kiểu dữ liệu NumPy)
        action = int(action)

        next_state, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        rewards.append(reward)
        action_history.append(action)

        state = next_state
        done = terminated or truncated
        steps += 1

    # Tính toán độ ổn định (Switching rate)
    switching_rate = np.mean(
        np.array(action_history[1:]) != np.array(action_history[:-1])
    )

    return {
        "total_reward": total_reward,
        "avg_reward": np.mean(rewards),
        "switching_rate": switching_rate,
        "steps": steps,
        "actions": action_history  
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
        dqn.q_net.load_state_dict(torch.load(DQN_MODEL_PATH, map_location=torch.device('cpu')))
        # dqn.q_net.load_state_dict(torch.load(DQN_MODEL_PATH))
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
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"evaluation_results_{timestamp}.csv"
    results_csv_path = os.path.join(RESULTS_DIR, csv_filename)
    results_df.to_csv(results_csv_path, index=False)
    print(f"Đã lưu kết quả đánh giá vào: {results_csv_path}")


    # In kết quả tổng hợp
    print("\n" + "="*45)
    print(f"{'MODEL':<15} | {'REWARD':<10} | {'SWITCH RATE':<12}")
    print("-" * 45)
    for r in all_results:
        print(f"{r['model']:<15} | {r['total_reward']:<10.2f} | {r['switching_rate']:<12.4f}")
    print("="*45)
    
if __name__ == "__main__":
    main()