import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_evaluation_charts():
    # Cài đặt giao diện chuẩn báo cáo khoa học (Whitegrid, font chữ to rõ)
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12})
    
    # Tạo một khung hình (Figure) chứa 3 biểu đồ con (1 hàng, 3 cột)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # ==========================================
    # BIỂU ĐỒ 1: ĐƯỜNG CONG HỌC TẬP (LEARNING CURVE)
    # ==========================================
    ax1 = axes[0]
    csv_path = "../results/ppo_multi_seed_results.csv"
    
    # Đọc dữ liệu từ file CSV đa hạt giống (Nếu có)
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        episodes = df['episode']
        mean_reward = df['mean_reward']
        std_reward = df['std_reward']
    else:
        # Tự động tạo dữ liệu mô phỏng để test nếu bạn chưa chạy file multi-seed
        print("Không tìm thấy file CSV multi-seed, đang dùng dữ liệu mô phỏng cho biểu đồ 1.")
        episodes = np.arange(0, 3000, 100)
        mean_reward = 1000 - 1500 * np.exp(-episodes / 500)
        std_reward = 100 * np.exp(-episodes / 1000)

    # Vẽ đường trung bình (Mean)
    ax1.plot(episodes, mean_reward, label='PPO Mean Reward', color='blue', linewidth=2)
    # Vẽ vùng đổ bóng (Độ lệch chuẩn)
    ax1.fill_between(episodes, mean_reward - std_reward, mean_reward + std_reward, 
                     color='blue', alpha=0.2, label='± 1 Std Dev')
    
    ax1.set_title("1. PPO Learning Curve (Multi-seed)")
    ax1.set_xlabel("Episodes")
    ax1.set_ylabel("Total Reward")
    ax1.legend(loc="lower right")

    # ==========================================
    # BIỂU ĐỒ 2: SO SÁNH TỔNG PHẦN THƯỞNG
    # ==========================================
    ax2 = axes[1]
    # Dữ liệu thực tế từ lần đánh giá offline của bạn
    models = ['DQN', 'Rule-based', 'PPO']
    rewards = [-708.8, 916.6, 1000.0]
    colors_reward = ['#e74c3c', '#f39c12', '#2ecc71'] # Đỏ (Xấu), Vàng (Tạm), Xanh (Tốt)

    bars1 = ax2.bar(models, rewards, color=colors_reward, width=0.6)
    ax2.set_title("2. Offline Evaluation: Total Reward")
    ax2.set_ylabel("Reward Score")
    ax2.axhline(0, color='black', linewidth=1) # Kẻ vạch số 0 cho dễ nhìn
    
    # Hiển thị số liệu trực tiếp trên đỉnh cột
    for bar in bars1:
        yval = bar.get_height()
        offset = 20 if yval > 0 else -40
        ax2.text(bar.get_x() + bar.get_width()/2, yval + offset, 
                 f'{yval:.1f}', ha='center', va='bottom', fontweight='bold')

    # ==========================================
    # BIỂU ĐỒ 3: SO SÁNH TỶ LỆ FLAPPING (SWITCHING RATE)
    # ==========================================
    ax3 = axes[2]
    # Đổi tỷ lệ thập phân sang phần trăm (%)
    switching_rates = [80.78, 7.4, 0.0] 
    colors_switch = ['#e74c3c', '#f39c12', '#3498db'] # DQN đỏ, Rule vàng, PPO xanh dương

    bars2 = ax3.bar(models, switching_rates, color=colors_switch, width=0.6)
    ax3.set_title("3. System Stability: Switching Rate (%)")
    ax3.set_ylabel("Rule Flapping (%)")
    ax3.set_ylim(0, 100) # Cố định trục Y từ 0 đến 100%
    
    # Hiển thị phần trăm trực tiếp trên đỉnh cột
    for bar in bars2:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, yval + 2, 
                 f'{yval:.2f}%', ha='center', va='bottom', fontweight='bold')

    # ==========================================
    # LƯU VÀ HIỂN THỊ
    # ==========================================
    plt.tight_layout()
    os.makedirs("../results", exist_ok=True)
    save_path = "../results/evaluation_charts.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight') # Lưu ảnh độ phân giải cao
    print(f"Đã xuất biểu đồ thành công tại: {save_path}")
    
    # Bật cửa sổ xem ảnh nếu chạy trên máy tính cá nhân
    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    generate_evaluation_charts()