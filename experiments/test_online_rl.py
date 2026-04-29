# experiments/test_online_rl.py

from rl_engine.online_env import OnlineSDNEnv
from rl_engine.agent.dqn_agent import DQNAgent
from rl_engine.agent.ppo_agent import PPOAgent
from rl_engine.config import STATE_DIM, ACTION_DIM


env = OnlineSDNEnv(
    controller_url="http://34.126.64.185:8181/onos/v1"
)

# ===== Load DQN =====
dqn_agent = DQNAgent(STATE_DIM, ACTION_DIM)
dqn_agent.load("models/train_dqn.pth")
dqn_agent.epsilon = 0.0

# ===== Load PPO =====
ppo_agent = PPOAgent(STATE_DIM, ACTION_DIM)
ppo_agent.load("models/train_ppo.pth")

print("\n========== DQN ONLINE TEST ==========\n")

state = env.reset()

for step in range(10):
    action = dqn_agent.select_action(state)
    next_state, reward, done, _ = env.step(action)

    print(f"[DQN] Step {step} | Action: {action} | Reward: {reward}")
    state = next_state


print("\n========== PPO ONLINE TEST ==========\n")

state = env.reset()

for step in range(10):
    action = ppo_agent.select_action(state)
    next_state, reward, done, _ = env.step(action)

    print(f"[PPO] Step {step} | Action: {action} | Reward: {reward}")
    state = next_state