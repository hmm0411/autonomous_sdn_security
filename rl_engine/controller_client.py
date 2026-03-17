import requests
import numpy as np

class ControllerClient:
    def __init__(self, controller_url="http://controller:8080", use_mock=True):
        self.controller_url = controller_url
        self.use_mock = use_mock


    def get_state(self):
        if self.use_mock:
            return {
                "total_packets": np.random.randint(100, 1000),
                "total_bytes": np.random.randint(1000, 10000),
                "flow_count": np.random.randint(10, 100),
                "latency": np.random.rand(),
                "packet_loss": np.random.rand(),
                "attack_indicator": np.random.randint(0, 2)
            }
        
        r = requests.get(f"{self.controller_url}/state")
        return r.json()

    def apply_action(self, action):
        if self.use_mock:
            # Return a mock response for testing purposes
            return {"status": "mock_action"}
        r = requests.post(
            f"{self.controller_url}/apply_action",
            json={"action": action}
        )
        return r.json()