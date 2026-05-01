from prometheus_client import start_http_server, Gauge

# ===== CORE =====
reward_g = Gauge('rl_reward', 'Reward', ['agent'])
latency_g = Gauge('rl_latency', 'Latency', ['agent'])
packet_loss_g = Gauge('rl_packet_loss', 'Packet loss', ['agent'])
flow_g = Gauge('rl_flow_count', 'Flow count', ['agent'])

# ===== ACTION =====
action_g = Gauge('rl_action', 'Action', ['agent'])

# ===== SYSTEM =====
cpu_g = Gauge('rl_controller_cpu', 'Controller CPU', ['agent'])
queue_g = Gauge('rl_queue_length', 'Queue length', ['agent'])

# ===== MODEL SELECTION =====
model_selected_g = Gauge('rl_model_selected', 'Selected model')

start_http_server(9100)

def update_metrics(state, reward, agent, action):
    reward_g.labels(agent=agent).set(reward)
    latency_g.labels(agent=agent).set(state[3])
    packet_loss_g.labels(agent=agent).set(state[4])
    flow_g.labels(agent=agent).set(state[2])

    action_g.labels(agent=agent).set(action)
    cpu_g.labels(agent=agent).set(state[7])
    queue_g.labels(agent=agent).set(state[6])
