# môi trường mô phỏng RL --> Đóng vai trò như một môi trường OpenAI Gym
# Action -> áp vào hệ thống -> Quan sát kết quả (reward) -> Cập nhật policy
import requests
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from rl_engine import reward
from rl_engine.controller_client import ControllerClient
from rl_engine.state_builder import StateBuilder
from rl_engine.reward import Reward
from rl_engine.config import STATE_DIM, ACTION_DIM

class SDNEnv(gym.Env):

    def __init__(self, controller_url="http://controller:8080"):
        self.controller = ControllerClient(controller_url=controller_url)
        self.state_builder = StateBuilder()
        self.reward_calc = Reward()

        self.action_space = spaces.Discrete(ACTION_DIM)
        self.observation_space = spaces.Box(low=0, high=1, shape=(STATE_DIM,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        raw = self.controller.get_state()
        state = self.state_builder.build(raw)
        return state, {}

    def step(self, action):
        self.controller.apply_action(action)

        raw = self.controller.get_state()
        state = self.state_builder.build(raw)
        reward = self.reward_calc.calculate(raw, action)

        terminated = False
        truncated = False
        info = raw

        return state, reward, terminated, truncated, info