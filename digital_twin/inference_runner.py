import os
import time
from xml.parsers.expat import model
import torch
import numpy as np
import torch.nn as nn
import random
from collector import ONOSCollector
from controller_client import ControllerClient
from twin import DigitalTwin
from safety import validate
from transition_logger import TransitionLogger

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DQN_PATH = os.path.join(BASE_DIR, 'models', 'dqn_model.pth')
MODEL_PPO_PATH = os.path.join(BASE_DIR, 'models', 'ppo_model.pth')
ACTIVE_MODEL = "DQN"  # Hoặc "PPO"
ATTACK_TYPE = "ddos"

collector = ONOSCollector()
controller = ControllerClient()
# twin = DigitalTwin("../../models/surrogate_model.pkl")
logger = TransitionLogger("transition_dataset.csv")

class QNetwork(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

def load_model():
    # if ACTIVE_MODEL == "DQN":
    #     model = torch.load(MODEL_DQN_PATH)
    # elif ACTIVE_MODEL == "PPO":
    #     model = torch.load(MODEL_PPO_PATH)
    # else:
    #     raise ValueError("ACTIVE_MODEL phải là 'DQN' hoặc 'PPO'")

    # model.eval()
    # return model
    state_dim = 9  # Số lượng features của state
    action_dim = 5  # Số lượng action (ví dụ: 0,1,2)
    model = QNetwork(state_dim, action_dim)
    state_dict = torch.load(MODEL_DQN_PATH)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def select_action(model, state):
    # Đảm bảo state được chuẩn hóa trước khi đưa vào PyTorch
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        action = model(state_tensor).argmax().item()
    return action

# def run():
#     model = load_model()
#     print("Khởi động hệ thống phòng vệ tự động SDN...")

#     while True:
#         try:
#             # 1. Lấy trạng thái từ mạng thật (Mininet -> ONOS)
#             state = collector.get_state()
#             if state is None:
#                 continue

#             # 2. RL Suy luận hành động
#             action = select_action(model, state)
#             print(f"RL đề xuất hành động: {action}")

#             # 3. Twin Mô phỏng (Policy Validator)
#             twin.update_state(state)
#             predicted = twin.simulate(action)
#             print(f"Twin dự đoán QoS: {predicted}")

#             # 4. Kiểm tra an toàn
#             if validate(predicted):
#                 print("Twin ACCEPT: Đang áp dụng luật xuống SDN...")
#                 controller.apply_action(action)
                
#                 # Chờ mạng hội tụ trạng thái mới
#                 time.sleep(2) 
#                 next_state = collector.get_state()
                
#                 # Ghi log làm dữ liệu đánh giá
#                 logger.log(state, action, next_state, ATTACK_TYPE)
#             else:
#                 print("Twin REJECT: Hành động bị hủy do rủi ro mạng.")

#             time.sleep(3)
            
#         except Exception as e:
#             print(f"Lỗi hệ thống: {e}")
#             time.sleep(5)

def run():
    model = load_model()
    print("=== CHẾ ĐỘ THU THẬP DỮ LIỆU CHO DIGITAL TWIN ===")
    
    while True:
        try:
            # 1. Lấy trạng thái mạng
            state = collector.get_state()
            if state is None:
                continue

            # 2. RL Ra quyết định (Lấy từ model)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()
            
            # Để Twin học được nhiều tình huống, thỉnh thoảng (20%) ta ép nó làm 
            # hành động ngẫu nhiên để xem mạng bị ảnh hưởng thế nào (Exploration)
            if random.random() < 0.2:
                action = random.choice([0, 1, 2, 3, 4]) # Giả sử bạn có 5 action: 0,1,2,3,4

            print(f"[RL] Hành động được chọn: {action}. Đang áp dụng xuống ONOS...")

            # 3. ÉP THỰC THI THẲNG XUỐNG MẠNG (BỎ QUA TWIN)
            controller.apply_action(action)
            
            # 4. Chờ mạng phản ứng và tính toán độ trễ, mất gói
            time.sleep(3) 
            next_state = collector.get_state()
            
            # 5. Ghi log vào file CSV
            # Lưu ý: Cột cuối cùng (Attack Type) hiện để tạm là 'mixed'
            logger.log(state, action, next_state, "mixed")
            print(f"Đã ghi Log: Action={action} | Next Latency={next_state[4]}ms | Next Loss={next_state[5]}")
            
            time.sleep(2)
            
        except Exception as e:
            print(f"Lỗi: {e}")
            time.sleep(5)

if __name__ == "__main__":
    run()