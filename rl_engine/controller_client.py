class ControllerClient:
    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove any attributes that cannot be pickled
        if 'socket' in state:
            del state['socket']
        return state
    def apply_action(self, action):
        # Send the action to the controller
        self.socket.send(action.encode())
        # Wait for the response from the controller
        response = self.socket.recv(1024).decode()
        return response
    