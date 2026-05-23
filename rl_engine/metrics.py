# rl_engine/metrics.py
from prometheus_client import Gauge

# Sử dụng chung 1 bộ metric, phân biệt bằng label 'model_type'
EPISODE_REWARD = Gauge('episode_reward', 'Phần thưởng thô', ['model_type'])
EPISODE_REWARD_MEAN = Gauge('episode_reward_mean', 'Phần thưởng trung bình', ['model_type'])
EPISODE_REWARD_STD = Gauge('episode_reward_std', 'Độ lệch chuẩn phần thưởng', ['model_type'])
EPISODE_REWARD_BEST = Gauge('episode_reward_best', 'Kỷ lục tốt nhất', ['model_type'])
TRAINING_LOSS = Gauge('training_loss', 'Loss của mô hình', ['model_type'])