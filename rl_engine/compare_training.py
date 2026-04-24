# File: compare_training.py (Chạy từ thư mục gốc: python compare_training.py)
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_learning_curves():
    # Dựa vào thư mục xuất file của log bạn, nó nằm ở folder results/ góc
    dqn_csv = "results/dqn_summary_results.csv"
    ppo_csv = "results/ppo_summary_results.csv"
    
    # Cấu hình giao diện chuẩn
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))
    
    # 1. Vẽ đường hội tụ DQN
    if os.path.exists(dqn_csv):
        df_dqn = pd.read_csv(dqn_csv)
        plt.plot(df_dqn['episode'], df_dqn['mean_reward'], label='DQN Mean Reward', color='#e74c3c') # Đỏ
        plt.fill_between(df_dqn['episode'], 
                         df_dqn['mean_reward'] - df_dqn['std_reward'], 
                         df_dqn['mean_reward'] + df_dqn['std_reward'], 
                         color='#e74c3c', alpha=0.2)
    else:
        print(f"Lỗi: Không tìm thấy file {dqn_csv}")

    # 2. Vẽ đường hội tụ PPO
    if os.path.exists(ppo_csv):
        df_ppo = pd.read_csv(ppo_csv)
        plt.plot(df_ppo['episode'], df_ppo['mean_reward'], label='PPO Mean Reward', color='#3498db') # Xanh dương
        plt.fill_between(df_ppo['episode'], 
                         df_ppo['mean_reward'] - df_ppo['std_reward'], 
                         df_ppo['mean_reward'] + df_ppo['std_reward'], 
                         color='#3498db', alpha=0.2)
    else:
        print(f"Lỗi: Không tìm thấy file {ppo_csv}")

    # Tuỳ biến Biểu đồ
    plt.title("So sánh Reward trong quá trình huấn luyện: DQN vs PPO")
    plt.xlabel("Trận/Tập (Episode)")
    plt.ylabel("Trung bình Reward (Tích lũy)")
    plt.legend()
    plt.tight_layout()
    
    # Xuất ảnh
    os.makedirs("results", exist_ok=True)
    out_path = "results/compare_learning_curves.png"
    plt.savefig(out_path, dpi=300)
    print(f"Đã xuất sơ đồ thành công tại: {out_path}")
    
    plt.show()

if __name__ == "__main__":
    plot_learning_curves()
