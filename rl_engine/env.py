# môi trường mô phỏng RL --> Đóng vai trò như một môi trường OpenAI Gym
# Action -> áp vào hệ thống -> Quan sát kết quả (reward) -> Cập nhật policy
import requests
import gym
from gym import spaces
import numpy as np
from rl_engine.controller_client import ControllerClient
from rl_engine.state_builder import StateBuilder
from rl_engine.reward_calculator import RewardCalculator
from rl_engine.config import STATE_DIM, ACTION_DIM

class SDNEnv(gym.Env):

    def __init__(self):
        self.controller = ControllerClient()
        self.state_builder = StateBuilder()
        self.reward_calc = RewardCalculator()

        self.action_space = spaces.Discrete(ACTION_DIM)
        self.observation_space = spaces.Box(low=0, high=1, shape=(STATE_DIM,), dtype=np.float32)

    def get_state(self):
        r = requests.get("http://controller:8080/state")
        return r.json()

    def reset(self):
        raw = self.controller.get_state()
        state = self.state_builder.build(raw)
        return state

    def step(self, action): 
        self.controller.apply_action(action)
        raw = self.controller.get_state()
        state = self.state_builder.build(raw)
        reward = self.reward_calc.calculate(raw, action)

        done = False
        info = raw
        return state, reward, done, info