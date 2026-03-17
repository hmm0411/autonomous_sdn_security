"""
baseline_rule.py

New state format:
S = [
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
    byte_rate_threshold: float = 500000.0
    flow_count_threshold: int = 100
    src_ip_entropy_threshold: float = 1.5
    latency_threshold: float = 100.0
    packet_loss_threshold: float = 0.1
    queue_length_threshold: float = 50.0
    controller_cpu_threshold: float = 0.8

    severe_packet_rate_threshold: float = 3000.0
    severe_byte_rate_threshold: float = 2000000.0
    severe_flow_count_threshold: int = 300
    severe_latency_threshold: float = 300.0
    severe_packet_loss_threshold: float = 0.3
    severe_queue_length_threshold: float = 200.0
    severe_cpu_threshold: float = 0.9

    no_action_id: int = 0
    block_action_id: int = 1
    limit_bw_action_id: int = 2
    redirect_action_id: int = 3
    isolate_action_id: int = 4


class BaselineRuleBasedAgent:
    """
    Expected state format:
    state = [
        packet_rate,
        byte_rate,
        flow_count,
        src_ip_entropy,
        latency,
        packet_loss,
        queue_length,
        controller_cpu,
        attack_indicator,
        previous_action
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

        packet_rate = parsed["packet_rate"]
        byte_rate = parsed["byte_rate"]
        flow_count = parsed["flow_count"]
        src_ip_entropy = parsed["src_ip_entropy"]
        latency = parsed["latency"]
        packet_loss = parsed["packet_loss"]
        queue_length = parsed["queue_length"]
        controller_cpu = parsed["controller_cpu"]
        attack_indicator = parsed["attack_indicator"]
        previous_action = parsed["previous_action"]

        # =========================
        # Rule-based decision logic
        # =========================

        # Rule 1: Severe attack / system overload -> isolate
        if (
            attack_indicator == 1
            and (
                packet_rate > self.config.severe_packet_rate_threshold
                or byte_rate > self.config.severe_byte_rate_threshold
                or flow_count > self.config.severe_flow_count_threshold
                or latency > self.config.severe_latency_threshold
                or packet_loss > self.config.severe_packet_loss_threshold
                or queue_length > self.config.severe_queue_length_threshold
                or controller_cpu > self.config.severe_cpu_threshold
            )
        ):
            action = self.config.isolate_action_id

        # Rule 2: High-confidence attack symptoms -> block
        elif (
            attack_indicator == 1
            and (
                packet_rate > self.config.packet_rate_threshold
                or flow_count > self.config.flow_count_threshold
                or src_ip_entropy > self.config.src_ip_entropy_threshold
            )
        ):
            action = self.config.block_action_id

        # Rule 3: Congestion/network degradation -> limit bandwidth
        elif (
            byte_rate > self.config.byte_rate_threshold
            or queue_length > self.config.queue_length_threshold
            or packet_loss > self.config.packet_loss_threshold
        ):
            action = self.config.limit_bw_action_id

        # Rule 4: Latency high but not yet severe -> redirect
        elif latency > self.config.latency_threshold:
            action = self.config.redirect_action_id

        # Rule 5: Avoid overly aggressive repeated isolate/block
        elif previous_action == self.config.isolate_action_id and attack_indicator == 0:
            action = self.config.no_action_id

        elif previous_action == self.config.block_action_id and attack_indicator == 0:
            action = self.config.no_action_id

        # Default: no action
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
                "packet_rate": float(state.get("packet_rate", 0.0)),
                "byte_rate": float(state.get("byte_rate", 0.0)),
                "flow_count": int(state.get("flow_count", 0)),
                "src_ip_entropy": float(state.get("src_ip_entropy", 0.0)),
                "latency": float(state.get("latency", 0.0)),
                "packet_loss": float(state.get("packet_loss", 0.0)),
                "queue_length": float(state.get("queue_length", 0.0)),
                "controller_cpu": float(state.get("controller_cpu", 0.0)),
                "attack_indicator": int(state.get("attack_indicator", 0)),
                "previous_action": int(state.get("previous_action", self.last_action)),
            }

        if isinstance(state, (list, tuple, np.ndarray)):
            if len(state) < 10:
                raise ValueError(
                    f"State must have at least 10 elements, got {len(state)}"
                )

            return {
                "packet_rate": float(state[0]),
                "byte_rate": float(state[1]),
                "flow_count": int(state[2]),
                "src_ip_entropy": float(state[3]),
                "latency": float(state[4]),
                "packet_loss": float(state[5]),
                "queue_length": float(state[6]),
                "controller_cpu": float(state[7]),
                "attack_indicator": int(state[8]),
                "previous_action": int(state[9]),
            }

        raise TypeError(
            "Unsupported state type. Expected dict, list, tuple, or np.ndarray."
        )


def block_flow(controller, state):
    return controller.apply_action(1, state=state)


def no_action(controller, state):
    return controller.apply_action(0, state=state)


def limit_bandwidth(controller, state):
    return controller.apply_action(2, state=state)


def redirect_traffic(controller, state):
    return controller.apply_action(3, state=state)


def isolate_device(controller, state):
    return controller.apply_action(4, state=state)


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