import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional

from rl_engine.config import GAMMA, LR


ACTION_MAP = {
    0: "no_action",
    1: "block_suspicious_flow",
    2: "limit_bandwidth",
    3: "redirect_traffic",
    4: "isolate_device",
}


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.actor = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Linear(128, 1)

    def forward(self, state):
        features = self.shared(state)
        policy = self.actor(features)
        value = self.critic(features)
        return policy, value


class PPOAgent:
    """
    Expected state format:
    state = [
        packet_rate,        # 0
        byte_rate,          # 1
        flow_count,         # 2
        src_ip_entropy,     # 3
        latency,            # 4
        packet_loss,        # 5
        queue_length,       # 6
        controller_cpu,     # 7
        attack_indicator,   # 8
        previous_action     # 9
    ]
    """

    def __init__(
        self,
        state_dim: int = 10,
        action_dim: int = 5,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01, # tăng lên 0.05 nếu lúc nào agent cũng chọn đúng Action 1 mà không chịu dùng Action 4 dù mạng vẫn đang
        value_coef: float = 0.5,
        device: Optional[str] = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)

        self.last_action = 0

    def reset(self):
        self.last_action = 0

    def _parse_state(self, state) -> np.ndarray:
        if isinstance(state, dict):
            return np.array([
                float(state.get("packet_rate", 0.0)),
                float(state.get("byte_rate", 0.0)),
                float(state.get("flow_count", 0)),
                float(state.get("src_ip_entropy", 0.0)),
                float(state.get("latency", 0.0)),
                float(state.get("packet_loss", 0.0)),
                float(state.get("queue_length", 0.0)),
                float(state.get("controller_cpu", 0.0)),
                float(state.get("attack_indicator", 0)),
                float(state.get("previous_action", self.last_action)),
            ], dtype=np.float32)

        if isinstance(state, (list, tuple, np.ndarray)):
            if len(state) < self.state_dim:
                raise ValueError(
                    f"State must have at least {self.state_dim} elements, got {len(state)}"
                )
            return np.array(state[:self.state_dim], dtype=np.float32)

        raise TypeError(
            "Unsupported state type. Expected dict, list, tuple, or np.ndarray."
        )

    def select_action(self, state):
        """
        Stochastic action selection for training
        """
        state = self._parse_state(state)
        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            policy, _ = self.model(state_tensor)

        dist = torch.distributions.Categorical(policy)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        self.last_action = action.item()
        return action.item(), log_prob.item()

    def select_greedy_action(self, state):
        """
        Deterministic action selection for evaluation/inference
        """
        state = self._parse_state(state)
        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            policy, _ = self.model(state_tensor)

        action = torch.argmax(policy, dim=-1).item()
        self.last_action = action
        return action

    def predict(self, state) -> int:
        return self.select_greedy_action(state)

    def compute_returns(self, rewards, dones):
        returns = []
        R = 0.0

        for r, d in zip(reversed(rewards), reversed(dones)):
            if d:
                R = 0.0
            R = r + GAMMA * R
            returns.insert(0, R)

        return torch.tensor(returns, dtype=torch.float32, device=self.device)

    def update(self, states, actions, log_probs_old, rewards, dones):
        if len(states) == 0:
            raise ValueError("Empty batch passed to update().")

        states_np = np.array([self._parse_state(s) for s in states], dtype=np.float32)

        states = torch.tensor(states_np, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        log_probs_old = torch.tensor(log_probs_old, dtype=torch.float32, device=self.device)

        returns = self.compute_returns(rewards, dones)

        policy, values = self.model(states)
        values = values.squeeze(-1)

        dist = torch.distributions.Categorical(policy)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ratios = torch.exp(log_probs - log_probs_old)

        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages

        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = nn.MSELoss()(values, returns)

        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
        }

    def save(self, path: str):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "last_action": self.last_action,
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.last_action = checkpoint.get("last_action", 0)
        self.model.to(self.device)