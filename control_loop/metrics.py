from prometheus_client import start_http_server, Gauge

reward_g = Gauge('rl_reward', 'Reward', ['model_type', 'stage'])
latency_g = Gauge('rl_latency', 'Latency', ['model_type'])
packet_loss_g = Gauge('rl_packet_loss', 'Packet loss', ['model_type'])
flow_g = Gauge('rl_flow_count', 'Flow count', ['model_type'])

action_g = Gauge('rl_action', 'Action', ['model_type'])
cpu_g = Gauge('rl_controller_cpu', 'Controller CPU', ['model_type'])
queue_g = Gauge('rl_queue_length', 'Queue length', ['model_type'])

model_selected_g = Gauge('rl_model_selected', 'Selected model', ['model_type'])

start_http_server(9100)

def update_metrics(state, reward_prod, reward_staging, model_type_str, action):
    # Đã sửa sạch sẽ thành model_type=...
    reward_g.labels(model_type=model_type_str, stage='production').set(reward_prod)
    reward_g.labels(model_type=model_type_str, stage='staging').set(reward_staging)

    latency_g.labels(model_type=model_type_str).set(state[5]) 
    packet_loss_g.labels(model_type=model_type_str).set(state[6]) 
    flow_g.labels(model_type=model_type_str).set(state[2])

    action_g.labels(model_type=model_type_str).set(action)
    cpu_g.labels(model_type=model_type_str).set(state[7])

    model_selected_g.labels(model_type=model_type_str).set(1)