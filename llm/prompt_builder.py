import numpy as np

def build_prompt(state, action, qos):
    return f"""
You are a senior SDN network security expert.

The network is operating under potential attack conditions.

Network state (abstracted features):
{np.round(state, 3).tolist()}

Action selected by RL agent:
{action}

QoS metrics after action:
- Delay: {qos.get('delay', 'unknown')} ms
- Packet loss: {qos.get('loss', 'unknown')} %
- Throughput: {qos.get('throughput', 'unknown')} Mbps

Tasks:
1. Explain why this action was selected based on the network state
2. Evaluate its impact on QoS
3. Determine whether the action is SAFE or RISKY
4. Provide a concise explanation (3-5 sentences)

Output format:
Safety: SAFE or RISKY
Reason: ...
QoS Impact: ...
Explanation: ...
"""