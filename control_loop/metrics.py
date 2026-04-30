from prometheus_client import Gauge
import numpy as np

# Khởi tạo các biểu đồ cho Prometheus
RL_REWARD = Gauge('rl_current_reward', 'Current reward from RL agent')
PACKET_RATE = Gauge('sdn_packet_rate', 'Current packet rate in network')
ACTION_TAKEN = Gauge('rl_action_taken', 'Action selected by RL agent')

def update_metrics(state, reward):
    """
    Hàm đẩy dữ liệu trạng thái và phần thưởng lên Prometheus
    """
    try:
        # Cập nhật Reward
        RL_REWARD.set(float(reward))
        
        # Cập nhật Action 
        action_val = state[-1] if isinstance(state, (list, np.ndarray)) else 0
        ACTION_TAKEN.set(float(action_val))
        
        # Cập nhật Packet Rate 
        packet_rate_val = state[0] if isinstance(state, (list, np.ndarray)) else 0
        PACKET_RATE.set(float(packet_rate_val))
        
    except Exception as e:
        print(f"Lỗi khi cập nhật Metrics: {e}")