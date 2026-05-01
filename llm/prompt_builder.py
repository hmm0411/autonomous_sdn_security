import numpy as np

def build_prompt(state, action, qos):
    # 1. Ánh xạ Action ID thành Tên hành động cụ thể
    action_map = {
        0: "No Action",
        1: "Block",
        2: "Limit Bandwidth",
        3: "Redirect",
        4: "Isolate"
    }
    action_name = action_map.get(action, f"Unknown Action ({action})")

    # 2. Gắn nhãn cho mảng State Vector 
    feature_names = [
        "Packet Rate", "Byte Rate", "Flow Count", "Src IP Entropy",
        "Latency", "Packet Loss", "Queue Length", "Controller CPU",
        "Attack Indicator", "Previous Action"
    ]
    
    # Chuyển đổi an toàn sang list và làm tròn
    state_values = np.round(state, 3).tolist() if hasattr(state, 'tolist') else [round(float(x), 3) for x in state]
    
    # Ghép tên đặc trưng với giá trị để LLM dễ đọc
    state_dict = dict(zip(feature_names, state_values))
    state_str = "\n".join([f"- {k}: {v}" for k, v in state_dict.items()])

    # 3. Xây dựng Prompt 
    return f"""
You are the "LLM Cognition Layer" of an Autonomous SDN Security System. 
Your task is to explain the decision made by the Reinforcement Learning (RL) agent to the human network operator.

[NETWORK CONTEXT]
Current Network State (Normalized):
{state_str}

RL Agent Decision:
- Executed Action: {action_name}

Post-Action QoS Metrics:
- Delay: {qos.get('delay', 'N/A')} ms
- Packet Loss: {qos.get('loss', 'N/A')} %
- Throughput: {qos.get('throughput', 'N/A')} Mbps

[TASKS]
1. Reason: Explain logically why '{action_name}' was triggered based on the specific metrics in the Current Network State (e.g., mention specific high values like Packet Rate or CPU).
2. QoS Impact: Assess how this action mitigates the threat and its effect on QoS metrics.
3. Safety: Classify the network status as SAFE (action mitigated the risk) or RISKY (threat persists or metrics are unstable).
4. Provide the explanation in a professional, concise manner suitable for a SOC (Security Operations Center) dashboard.

[OUTPUT FORMAT]
Safety: [SAFE or RISKY]
Reason: [Your technical explanation...]
QoS Impact: [Your QoS assessment...]
Explanation: [A concise 3-sentence summary for the operator]
"""