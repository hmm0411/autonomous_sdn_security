import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from rl_engine.config import GAMMA, LR_DQN, TARGET_UPDATE, ACTION_DIM, EPS_START


class QNetwork(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:

    def __init__(self, state_dim, action_dim):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)

        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LR_DQN)

        self.epsilon = EPS_START
        self.update_count = 0

    # =============================
    # ACTION SELECTION
    # =============================
    def select_action(self, state):

        if np.random.random() < self.epsilon:
            return np.random.randint(0, ACTION_DIM)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q = self.q_net(state_tensor)

        return q.argmax().item()

    # =============================
    # TRAIN UPDATE
    # =============================
    def update(self, batch):

        states, actions, rewards, next_states, dones = batch

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.q_net(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        next_q_values = self.target_net(next_states).max(1)[0]

        target = rewards + GAMMA * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_value, target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1

        if self.update_count % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    # =============================
    # LOAD MODEL (FOR ONLINE)
    # =============================
    def load(self, path: str) -> None:

        checkpoint = torch.load(path, map_location="cpu")

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # checkpoint dạng full
            self.q_net.load_state_dict(checkpoint["model_state_dict"])
            self.target_net.load_state_dict(checkpoint["target_model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.epsilon = checkpoint.get("epsilon", self.epsilon)
        else:
        # checkpoint chỉ là state_dict
            self.q_net.load_state_dict(checkpoint)
            self.target_net.load_state_dict(checkpoint)

        print("Model loaded successfully")