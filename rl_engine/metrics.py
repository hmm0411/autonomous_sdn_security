import random
from prometheus_client import Gauge

# Đổi label từ 'model_type' thành 'agent' để khớp với Dashboard hiện tại
EPISODE_REWARD = Gauge('episode_reward', 'Phần thưởng thô', ['agent'])
EPISODE_REWARD_MEAN = Gauge('episode_reward_mean', 'Phần thưởng trung bình', ['agent'])
EPISODE_REWARD_STD = Gauge('episode_reward_std', 'Độ lệch chuẩn phần thưởng', ['agent'])
EPISODE_REWARD_BEST = Gauge('episode_reward_best', 'Kỷ lục tốt nhất', ['agent'])
TRAINING_LOSS = Gauge('training_loss', 'Loss của mô hình', ['agent'])

RL_REWARD = EPISODE_REWARD # Alias để hàm simulate chạy được
RL_LOSS = TRAINING_LOSS     # Alias để hàm simulate chạy được
RL_EPSILON = Gauge('rl_epsilon', 'DQN Epsilon Decay (Exploration Rate)')

# --- METRIC HẠ TẦNG & TRẠNG THÁI ---
CONTROLLER_CPU = Gauge('controller_cpu_usage', 'CPU Usage of ONOS Controller')
CONTROLLER_FLOW_COUNT = Gauge('controller_flow_count', 'Number of Flows in ONOS Controller')
MODEL_STATUS = Gauge('model_status', 'Trạng thái deployment của model', ['stage', 'agent'])
CURRENT_ACTION = Gauge('rl_current_action_code', 'Mã hành động hiện tại', ['agent'])
THREAT_LEVEL = Gauge('threat_level', 'Mức độ đe dọa (0-100%)')

def simulate_training_metrics_if_idle(agent_type="dqn"):
    """
    Hàm giả lập dữ liệu: Đảm bảo Dashboard không trống dữ liệu khi hệ thống nhàn rỗi.
    """
    if agent_type == "dqn":
        RL_EPSILON.set(round(random.uniform(0.1, 0.2), 4))
        RL_REWARD.labels(agent='dqn').set(round(random.uniform(-50.0, -10.0), 2))
        RL_LOSS.labels(agent='dqn').set(round(random.uniform(0.01, 0.05), 4))
        MODEL_STATUS.labels(stage='production', agent='dqn').set(1.0)
        MODEL_STATUS.labels(stage='staging', agent='dqn').set(0.5)
    else:
        RL_REWARD.labels(agent='ppo').set(round(random.uniform(-40.0, -5.0), 2))
        RL_LOSS.labels(agent='ppo').set(round(random.uniform(0.005, 0.02), 4))
        MODEL_STATUS.labels(stage='production', agent='ppo').set(1.0)
        MODEL_STATUS.labels(stage='staging', agent='ppo').set(0.0)

    CONTROLLER_CPU.set(round(random.uniform(15.5, 45.8), 1))
    CONTROLLER_FLOW_COUNT.set(random.randint(100, 500))