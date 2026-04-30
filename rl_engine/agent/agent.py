# thêm agent.py để định nghĩa agent (DQN, PPO, v.v.) sẽ học từ môi trường SDNEnv và digital twin
import random
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from flask import Flask, request, jsonify
from rl_engine.config import STATE_DIM, ACTION_DIM
from rl_engine.logger import ppo_reward_gauge, dqn_reward_gauge

app = Flask(__name__)

# Lấy loại Agent từ môi trường
AGENT_TYPE = os.getenv('AGENT_TYPE', 'DQN').upper()

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

    def select_action(self, state):
        return random.choice([i for i in range(self.action_dim)])

    def update(self, state, action, reward, next_state):
        dqn_reward_gauge.set(reward)  # Cập nhật reward vào Gauge của DQN
        pass

class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

    def select_action(self, state):
        return random.choice([i for i in range(self.action_dim)])

    def update(self, state, action, reward, next_state):
        ppo_reward_gauge.set(reward)  # Cập nhật reward vào Gauge của PPO
        pass

if AGENT_TYPE == 'PPO':
    agent = PPOAgent(STATE_DIM, ACTION_DIM)
    print(f"Initialized PPO Agent API")
else:
    agent = DQNAgent(STATE_DIM, ACTION_DIM)
    print(f"Initialized DQN Agent API")

# SERVING API

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        state = data.get('state')
        action = agent.select_action(state)
        return jsonify({"action": action, "agent_type": AGENT_TYPE}), 200
    except Exception as e:
        return jsonify({"error": str(e), "action": 0}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 9000))
    app.run(host='0.0.0.0', port=port)