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
        
        # Giả sử các action có tác động như sau (đây chỉ là ví dụ, cần điều chỉnh theo thực tế):
        if action == "block":
            simulated_latency *= 0.8
        elif action == "limit":
            simulated_latency *= 0.9
        elif action == "redirect":
            simulated_latency *= 0.95
        elif action == "isolate":
            simulated_latency *= 0.7

        return simulated_latency