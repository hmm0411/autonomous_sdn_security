import requests

class ControllerClient:
    def __init__(self, controller_url):
        self.controller_url = 'http://controller:8080'

    def __getstate__(self):
        r = requests.get(f"{self.controller_url}/state")
        state = self.__dict__.copy()
        state = state.append(r.json())
        # Remove any attributes that cannot be pickled
        if 'socket' in state:
            del state['socket']
        return state
    def apply_action(self, action):
        # Send the action to the controller
        response = requests.post(f"{self.controller_url}/apply_action", json={"action": action})
        return response.json()
    