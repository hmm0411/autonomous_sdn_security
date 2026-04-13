import numpy as np
import joblib

class DigitalTwin:
    def __init__(self, model_path="../../models/surrogate_model.pkl"):
        self.model = joblib.load(model_path)
        self.current_state = None

    def update_state(self, state):
        # Đảm bảo state là một list/array có 9 phần tử
        self.current_state = np.array(state)

    def simulate(self, action):
        if self.current_state is None:
            return None

        # Ghép 9 features của state với 1 action -> 10 features khớp với lúc train
        x = np.concatenate([self.current_state, [action]])
        
        # Reshape thành mảng 2D cho sklearn predict
        prediction = self.model.predict(x.reshape(1, -1))[0]

        predicted_latency = prediction[0]
        predicted_loss = prediction[1]

        return {
            "latency": predicted_latency,
            "packet_loss": predicted_loss
        }