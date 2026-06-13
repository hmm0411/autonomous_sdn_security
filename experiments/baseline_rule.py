from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class RuleConfig:
    packet_rate_threshold: float = 1000.0
    byte_rate_threshold: float = 500000.0
    flow_count_threshold: float = 150.0
    flow_growth_threshold: float = 30.0
    src_ip_entropy_threshold: float = 2.0
    latency_threshold: float = 80.0
    packet_loss_threshold: float = 0.05
    controller_cpu_threshold: float = 80.0

    severe_packet_rate_threshold: float = 3000.0
    severe_flow_count_threshold: float = 300.0
    severe_flow_growth_threshold: float = 80.0
    severe_latency_threshold: float = 180.0
    severe_packet_loss_threshold: float = 0.2

    no_action_id: int = 0
    block_action_id: int = 1
    limit_bw_action_id: int = 2
    redirect_action_id: int = 3
    isolate_action_id: int = 4


class BaselineRuleBasedAgent:
    """
    Expected state 8:

    0 packet_rate
    1 byte_rate
    2 flow_count
    3 flow_growth_rate
    4 src_ip_entropy
    5 latency
    6 packet_loss
    7 controller_cpu
    """

    def __init__(self, config: Optional[RuleConfig] = None):
        self.config = config or RuleConfig()
        self.last_action = self.config.no_action_id
        self.action_history = []

    def reset(self):
        self.last_action = self.config.no_action_id
        self.action_history = []

    def predict(self, state) -> int:
        return self.act(state)

    def act(self, state) -> int:
        parsed = self._parse_state(state)

        packet_rate = parsed["packet_rate"]
        flow_count = parsed["flow_count"]
        flow_growth = abs(parsed["flow_growth_rate"])
        entropy = parsed["src_ip_entropy"]
        latency = parsed["latency"]
        packet_loss = parsed["packet_loss"]
        controller_cpu = parsed["controller_cpu"]

        if (
            packet_rate > self.config.severe_packet_rate_threshold
            or flow_count > self.config.severe_flow_count_threshold
            or packet_loss > self.config.severe_packet_loss_threshold
        ):
            action = self.config.block_action_id

        else:
            anomaly_score = 0

            if packet_rate > self.config.packet_rate_threshold:
                anomaly_score += 1

            if flow_count > self.config.flow_count_threshold:
                anomaly_score += 1

            if flow_growth > self.config.flow_growth_threshold:
                anomaly_score += 1

            if entropy > self.config.src_ip_entropy_threshold:
                anomaly_score += 1

            if latency > self.config.latency_threshold:
                anomaly_score += 1

            if packet_loss > self.config.packet_loss_threshold:
                anomaly_score += 1

            if controller_cpu > self.config.controller_cpu_threshold:
                anomaly_score += 1

            if anomaly_score >= 4:
                action = self.config.block_action_id
            elif anomaly_score >= 2:
                action = self.config.limit_bw_action_id
            elif entropy > 3.0 and flow_count > 80:
                action = self.config.redirect_action_id
            else:
                action = self.config.no_action_id

        self.last_action = int(action)
        self.action_history.append(int(action))

        return int(action)

    def _parse_state(self, state) -> Dict[str, Any]:
        if isinstance(state, dict):
            return {
                "packet_rate": float(state.get("packet_rate", 0.0)),
                "byte_rate": float(state.get("byte_rate", 0.0)),
                "flow_count": float(state.get("flow_count", 0.0)),
                "flow_growth_rate": float(state.get("flow_growth_rate", 0.0)),
                "src_ip_entropy": float(state.get("src_ip_entropy", 0.0)),
                "latency": float(state.get("latency", 0.0)),
                "packet_loss": float(state.get("packet_loss", 0.0)),
                "controller_cpu": float(state.get("controller_cpu", 0.0)),
            }

        if isinstance(state, (list, tuple, np.ndarray)):
            if len(state) < 8:
                raise ValueError(
                    f"State must have at least 8 elements, got {len(state)}"
                )

            return {
                "packet_rate": float(state[0]),
                "byte_rate": float(state[1]),
                "flow_count": float(state[2]),
                "flow_growth_rate": float(state[3]),
                "src_ip_entropy": float(state[4]),
                "latency": float(state[5]),
                "packet_loss": float(state[6]),
                "controller_cpu": float(state[7]),
            }

        raise TypeError(
            "Unsupported state type. Expected dict, list, tuple, or np.ndarray."
        )


def run_rule_based_episode(env, agent: BaselineRuleBasedAgent, max_steps: int = 1000):
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    step_count = 0

    agent.reset()

    while not done and step_count < max_steps:
        action = agent.act(obs)

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        done = terminated or truncated
        step_count += 1

    return {
        "episode_reward": total_reward,
        "steps": step_count,
        "action_history": agent.action_history.copy(),
    }