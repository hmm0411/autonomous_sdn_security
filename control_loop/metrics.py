from prometheus_client import start_http_server, Gauge

reward_g = Gauge("rl_reward", "Runtime reward", ["model_type", "stage"])

packet_rate_g = Gauge("rl_packet_rate", "Packet rate", ["model_type"])
byte_rate_g = Gauge("rl_byte_rate", "Byte rate", ["model_type"])
flow_growth_g = Gauge("rl_flow_growth_rate", "Flow growth rate", ["model_type"])

latency_g = Gauge("rl_latency", "Latency", ["model_type"])
packet_loss_g = Gauge("rl_packet_loss", "Packet loss", ["model_type"])
flow_g = Gauge("rl_flow_count", "Flow count", ["model_type"])
action_g = Gauge("rl_action", "Selected RL action", ["model_type"])
cpu_g = Gauge("rl_controller_cpu", "Controller CPU", ["model_type"])
queue_g = Gauge("rl_queue_length", "Queue length", ["model_type"])
model_selected_g = Gauge("rl_model_selected", "Selected model", ["model_type"])

start_http_server(9100)

def update_metrics(state, reward_prod, reward_staging, model_type_str, action):
    model_type_str = str(model_type_str).lower()

    reward_g.labels(model_type=model_type_str, stage="production").set(float(reward_prod))
    reward_g.labels(model_type=model_type_str, stage="staging").set(float(reward_staging))

    packet_rate_g.labels(model_type=model_type_str).set(float(state[0]))
    byte_rate_g.labels(model_type=model_type_str).set(float(state[1]))
    flow_g.labels(model_type=model_type_str).set(float(state[2]))
    flow_growth_g.labels(model_type=model_type_str).set(float(state[3]))

    latency_g.labels(model_type=model_type_str).set(float(state[5]))
    packet_loss_g.labels(model_type=model_type_str).set(float(state[6]))
    cpu_g.labels(model_type=model_type_str).set(float(state[7]))

    queue_g.labels(model_type=model_type_str).set(0.0)
    action_g.labels(model_type=model_type_str).set(int(action))
    model_selected_g.labels(model_type=model_type_str).set(1.0)