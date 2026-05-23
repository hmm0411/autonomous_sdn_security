# rl_engine/metrics.py
from prometheus_client import Gauge

# Sử dụng chung 1 bộ metric, phân biệt bằng label 'agent'
EPISODE_REWARD = Gauge('episode_reward', 'Phần thưởng thô của Episode', ['agent'])
EPISODE_REWARD_MEAN = Gauge('episode_reward_mean', 'Phần thưởng trung bình', ['agent'])
EPISODE_REWARD_STD = Gauge('episode_reward_std', 'Độ lệch chuẩn phần thưởng', ['agent'])
EPISODE_REWARD_BEST = Gauge('episode_reward_best', 'Kỷ lục tốt nhất', ['agent'])
TRAINING_LOSS = Gauge('training_loss', 'Loss của mô hình', ['agent'])