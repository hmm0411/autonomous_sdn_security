import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from rl_engine.config import GAMMA, LR, STATE_DIM, ACTION_DIM


class ActorCritic(nn.Module):

    def __init__(self):
        super(ActorCritic, self).__init__()

        # shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(STATE_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(128, ACTION_DIM),
            nn.Softmax(dim=-1)
        )

        # critic network (value function)
        self.critic = nn.Linear(128, 1)

    def forward(self, state):

        features = self.shared(state)

        policy = self.actor(features)
        value = self.critic(features)

        return policy, value


class PPOAgent:

    def __init__(self, state_dim, action_dim, clip_eps=0.2):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.clip_eps = clip_eps

        self.model = ActorCritic()

        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)

    def select_action(self, state):

        state = torch.FloatTensor(state).unsqueeze(0)

        policy, _ = self.model(state)

        dist = torch.distributions.Categorical(policy)

        action = dist.sample()

        log_prob = dist.log_prob(action)

        return action.item(), log_prob.item()

    def compute_returns(self, rewards, dones):

        returns = []
        R = 0

        for r, d in zip(reversed(rewards), reversed(dones)):

            if d:
                R = 0

            R = r + GAMMA * R
            returns.insert(0, R)

        return torch.FloatTensor(returns)

    def update(self, states, actions, log_probs_old, rewards, dones):

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        log_probs_old = torch.FloatTensor(log_probs_old)

        returns = self.compute_returns(rewards, dones)

        policy, values = self.model(states)

        values = values.squeeze()

        dist = torch.distributions.Categorical(policy)

        log_probs = dist.log_prob(actions)

        entropy = dist.entropy().mean()

        advantages = returns - values.detach()

        ratios = torch.exp(log_probs - log_probs_old)

        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages

        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = nn.MSELoss()(values, returns)

        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item()
        }