# experiments/test_online_rl.py
import pandas as pd

from rl_engine.online_env import OnlineSDNEnv
from rl_engine.agent.dqn_agent import DQNAgent
from rl_engine.agent.ppo_agent import PPOAgent
from rl_engine.config import STATE_DIM, ACTION_DIM


env = OnlineSDNEnv(
    controller_url="http://34.126.64.185:8181/onos/v1"
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
    next_state, reward, done, _ = env.step(action)

    timeline_dqn.append({
        "step": step,
        "flow_count": [next_state[2]],
        "entropy": [next_state[3]],
        "latency": [next_state[4]],
        "action": action,
        "reward": reward
    })
    print(f"[DQN] Step {step} | Action: {action} | Reward: {reward}" f"| Attack: {env._detect_attack(state) | env._detect_attack(next_state)}")
    state = next_state

df_dqn = pd.DataFrame(timeline_dqn)
df_dqn.to_csv("results/dqn_online_test.csv", index=False)

print("\n========== DQN ONLINE TEST COMPLETE ==========\n")


print("\n========== PPO ONLINE TEST ==========\n")

state = env.reset()

for step in range(50):
    action_tuple = ppo_agent.select_action(state)
    action = action_tuple[0] if isinstance(action_tuple, tuple) else action_tuple
    next_state, reward, done, _ = env.step(action)

    timeline_ppo.append({
        "step": step,
        "flow_count": [next_state[2]],
        "entropy": [next_state[3]],
        "latency": [next_state[4]],
        "action": action,
        "reward": reward
    })

    print(f"[PPO] Step {step} | Action: {action} | Reward: {reward} | Attack: {env._detect_attack(state) | env._detect_attack(next_state)}")
    state = next_state

df_ppo = pd.DataFrame(timeline_ppo)
df_ppo.to_csv("results/ppo_online_test.csv", index=False)

print("\n========== PPO ONLINE TEST COMPLETE ==========\n")