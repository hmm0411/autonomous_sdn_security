from prometheus_client import Gauge, start_http_server


reward_g = Gauge(
    "rl_reward",
    "Runtime reward",
    ["model_type", "stage"],
)

packet_rate_g = Gauge(
    "rl_packet_rate",
    "Packet rate",
    ["model_type"],
)

byte_rate_g = Gauge(
    "rl_byte_rate",
    "Byte rate",
    ["model_type"],
)

flow_count_g = Gauge(
    "rl_flow_count",
    "Flow count",
    ["model_type"],
)

flow_growth_g = Gauge(
    "rl_flow_growth_rate",
    "Flow growth rate",
    ["model_type"],
)

entropy_g = Gauge(
    "rl_src_ip_entropy",
    "Source IP entropy",
    ["model_type"],
)

latency_g = Gauge(
    "rl_latency",
    "Latency",
    ["model_type"],
)

packet_loss_g = Gauge(
    "rl_packet_loss",
    "Packet loss",
    ["model_type"],
)

cpu_g = Gauge(
    "rl_controller_cpu",
    "Controller CPU",
    ["model_type"],
)

action_g = Gauge(
    "rl_action",
    "Selected action",
    ["model_type"],
)

twin_safe_g = Gauge(
    "rl_twin_safe",
    "Digital Twin safety decision: 1=safe, 0=rejected",
    ["model_type"],
)

twin_gap_latency_g = Gauge(
    "rl_twin_gap_latency",
    "Digital Twin latency sim-to-real gap",
    ["model_type"],
)

model_selected_g = Gauge(
    "rl_model_selected",
    "Model selected",
    ["model_type"],
)


_started = False


def ensure_metrics_server(port=9100):
    global _started

    if not _started:
        start_http_server(port)
        _started = True
        print(f"[METRICS] Prometheus exporter started on port {port}", flush=True)


def update_metrics(
    state,
    reward_prod,
    reward_staging,
    model_type_str,
    action,
    twin_safe=1,
    twin_gap_latency=0.0,
):
    """
    State 8:
    0 packet_rate
    1 byte_rate
    2 flow_count
    3 flow_growth_rate
    4 src_ip_entropy
    5 latency
    6 packet_loss
    7 controller_cpu
    """
    model = str(model_type_str).lower()

    reward_g.labels(model_type=model, stage="production").set(float(reward_prod))
    reward_g.labels(model_type=model, stage="staging").set(float(reward_staging))

    packet_rate_g.labels(model_type=model).set(float(state[0]))
    byte_rate_g.labels(model_type=model).set(float(state[1]))
    flow_count_g.labels(model_type=model).set(float(state[2]))
    flow_growth_g.labels(model_type=model).set(float(state[3]))
    entropy_g.labels(model_type=model).set(float(state[4]))
    latency_g.labels(model_type=model).set(float(state[5]))
    packet_loss_g.labels(model_type=model).set(float(state[6]))
    cpu_g.labels(model_type=model).set(float(state[7]))

    action_g.labels(model_type=model).set(int(action))
    twin_safe_g.labels(model_type=model).set(int(twin_safe))
    twin_gap_latency_g.labels(model_type=model).set(float(twin_gap_latency))
    model_selected_g.labels(model_type=model).set(1.0)