# môi trường mô phỏng RL --> Đóng vai trò như một môi trường OpenAI Gym
# Action -> áp vào hệ thống -> Quan sát kết quả (reward) -> Cập nhật policy
import requests

class SDNEnv:

    def get_state(self):
        r = requests.get("http://controller:8080/state")
        return r.json()

    def reset(self):
        return self.get_state()

    def step(self, action):
        state = self.get_state()

        # reward đơn giản
        reward = -state["latency"]

        done = False
        return state, reward, done