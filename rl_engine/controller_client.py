import requests

class ControllerClient:
    def __init__(self, controller_url="http://controller:8080"):
        self.controller_url = controller_url

    def get_state(self):
        r = requests.get(f"{self.controller_url}/state")
        return r.json()

    def apply_action(self, action):
        r = requests.post(
            f"{self.controller_url}/apply_action",
            json={"action": action}
        )
        return r.json()