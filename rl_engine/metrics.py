from prometheus_client import Gauge

# Đổi label từ 'model_type' thành 'agent' để khớp với Dashboard hiện tại
EPISODE_REWARD = Gauge('episode_reward', 'Phần thưởng thô', ['agent'])
EPISODE_REWARD_MEAN = Gauge('episode_reward_mean', 'Phần thưởng trung bình', ['agent'])
EPISODE_REWARD_STD = Gauge('episode_reward_std', 'Độ lệch chuẩn phần thưởng', ['agent'])
EPISODE_REWARD_BEST = Gauge('episode_reward_best', 'Kỷ lục tốt nhất', ['agent'])
TRAINING_LOSS = Gauge('training_loss', 'Loss của mô hình', ['agent'])