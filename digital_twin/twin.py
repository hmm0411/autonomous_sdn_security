class DigitalTwin:
    def __init__(self):
        self.current_state = None

    def update_state(self, state):
        """
        Cập nhật state từ SDN thật
        """
        self.current_state = state

    def simulate(self, action):
        """
        Mô phỏng tác động của action
        Ví dụ: nếu block flow → giảm packet rate
        """
        if self.current_state is None:
            return None

        simulated_latency = self.current_state["latency"]

        if action == "block":
            simulated_latency *= 0.8
        elif action == "limit":
            simulated_latency *= 0.9

        return simulated_latency