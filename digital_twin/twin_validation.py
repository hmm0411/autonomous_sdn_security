import torch
import numpy as np
from digital_twin.train_surrogate import Surrogate

class TwinValidator:

    def __init__(self, model_path="models/surrogate.pth"):
        self.model = Surrogate()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def predict_next_state(self, state, action):
        x = np.append(state, action)
        x_tensor = torch.tensor(x).float().unsqueeze(0)

        with torch.no_grad():
            pred = self.model(x_tensor).numpy()[0]

        return pred

    def is_safe(self, predicted_state):
        latency = predicted_state[3]
        packet_loss = predicted_state[4]

        if latency > 0.8:
            return False
        if packet_loss > 0.5:
            return False

        return True