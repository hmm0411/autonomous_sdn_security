from prometheus_client import start_http_server, Gauge

from rl_engine.agent import agent


reward_g = Gauge('rl_reward', 'Reward', ['model_type', 'stage'])
latency_g = Gauge('rl_latency', 'Latency', ['model_type'])
packet_loss_g = Gauge('rl_packet_loss', 'Packet loss', ['model_type'])
flow_g = Gauge('rl_flow_count', 'Flow count', ['model_type'])

action_g = Gauge('rl_action', 'Action', ['model_type'])
cpu_g = Gauge('rl_controller_cpu', 'Controller CPU', ['model_type'])
queue_g = Gauge('rl_queue_length', 'Queue length', ['model_type'])

model_selected_g = Gauge('rl_model_selected', 'Selected model', ['model'])

start_http_server(9100)

def update_metrics(state, reward_prod, reward_staging, model_type, action):
    reward_g.labels(model=model_type, stage='production').set(reward_prod)
    reward_g.labels(model=model_type, stage='staging').set(reward_staging)

    latency_g.labels(model=model_type).set(state[5]) # Chuẩn 8-dim: latency index 5
    packet_loss_g.labels(model=model_type).set(state[6]) # Chuẩn 8-dim: loss index 6
    flow_g.labels(model=model_type).set(state[2])

    action_g.labels(model=model_type).set(action)
    cpu_g.labels(model=model_type).set(state[7])
    # queue_g.labels(model=model_type).set(state[6])

    model_selected_g.labels(model=model_type).set(1)