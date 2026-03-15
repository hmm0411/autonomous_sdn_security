"""
baseline_rule.py

Action mapping suggestion:
0 -> no_action
1 -> block_suspicious_flow
2 -> limit_bandwidth
3 -> redirect_traffic
4 -> isolate_device
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np


@dataclass
class RuleConfig:
    packet_rate_threshold: float = 1000.0

    alert_confidence_threshold: float = 0.7
    suspicious_flows_threshold: int = 5
    severe_packet_rate_threshold: float = 3000.0
    severe_cpu_threshold: float = 0.85

    
    no_action_id: int = 0
    block_action_id: int = 1
    limit_bw_action_id: int = 2
    redirect_action_id: int = 3
    isolate_action_id: int = 4


class BaselineRuleBasedAgent:
    """
    Expected state format:
    state = [
        attack_type,
        alert_confidence,
        packets_per_second,
        bytes_per_second,
        suspicious_flows,
        controller_cpu_load,
        latency,
        packet_loss
    ]
    """

    def __init__(self, config: Optional[RuleConfig] = None):
        self.config = config or RuleConfig()
        self.last_action = self.config.no_action_id
        self.action_history = []

    def reset(self):
        self.last_action = self.config.no_action_id
        self.action_history = []

    def act(self, state) -> int:
     

        parsed = self._parse_state(state)

        attack_type = parsed["attack_type"]
        alert_conf = parsed["alert_confidence"]
        pps = parsed["packets_per_second"]
        suspicious_flows = parsed["suspicious_flows"]
        cpu_load = parsed["controller_cpu_load"]

        # ---- Simple baseline ----
        # If packet rate exceeds threshold -> block
        if pps > self.config.packet_rate_threshold:
            action = self.config.block_action_id
        else:
            action = self.config.no_action_id

     
        self.last_action = action
        self.action_history.append(action)
        return action

    def predict(self, state) -> int:
        return self.act(state)

    def _parse_state(self, state) -> Dict[str, Any]:
    
        if isinstance(state, dict):
            return {
                "attack_type": state.get("attack_type", 0),
                "alert_confidence": float(state.get("alert_confidence", 0.0)),
                "packets_per_second": float(state.get("packets_per_second", 0.0)),
                "bytes_per_second": float(state.get("bytes_per_second", 0.0)),
                "suspicious_flows": int(state.get("suspicious_flows", 0)),
                "controller_cpu_load": float(state.get("controller_cpu_load", 0.0)),
                "latency": float(state.get("latency", 0.0)),
                "packet_loss": float(state.get("packet_loss", 0.0)),
            }

        if isinstance(state, (list, tuple, np.ndarray)):
            if len(state) < 8:
                raise ValueError(
                    f"State must have at least 8 elements, got {len(state)}"
                )
            return {
                "attack_type": state[0],
                "alert_confidence": float(state[1]),
                "packets_per_second": float(state[2]),
                "bytes_per_second": float(state[3]),
                "suspicious_flows": int(state[4]),
                "controller_cpu_load": float(state[5]),
                "latency": float(state[6]),
                "packet_loss": float(state[7]),
            }

        raise TypeError(
            "Unsupported state type. Expected dict, list, tuple, or np.ndarray."
        )


def block_flow(controller, state):
    return controller.apply_action(1, state=state)


def no_action(controller, state):
    return controller.apply_action(0, state=state)


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