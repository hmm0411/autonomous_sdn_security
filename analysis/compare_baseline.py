import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# --- Tự động tìm file CSV đánh giá mới nhất ---
list_of_files = glob.glob('results/evaluation_results_*.csv') 
latest_file = max(list_of_files, key=os.path.getctime)
print(f"Đang phân tích dữ liệu từ: {latest_file}")

# --- Chuẩn bị số liệu ---
df = pd.read_csv(latest_file)
models = df['model'].tolist()
rewards = df['total_reward'].tolist()
switching_rates = (df['switching_rate'] * 100).tolist() # Đổi ra % 

# Cấu hình giao diện biểu đồ
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- BIỂU ĐỒ 1: SO SÁNH TỔNG PHẦN THƯỞNG ---
axes[0].bar(models, rewards, color=['#e74c3c', '#3498db', '#f39c12'], width=0.5)
axes[0].axhline(0, color='black', linewidth=1)
axes[0].set_title('1. Tổng Phần Thưởng (Offline Evaluation)')
axes[0].set_ylabel('Reward Score')

for i, val in enumerate(rewards):
    offset = 5 if val > 0 else -15
    axes[0].text(i, val + offset, f'{val:.1f}', ha='center', va='bottom', fontweight='bold')

# --- BIỂU ĐỒ 2: SO SÁNH TỶ LỆ FLAP, MỨC ĐỘ ỔN ĐỊNH ---
axes[1].bar(models, switching_rates, color=['#e74c3c', '#3498db', '#f39c12'], width=0.5)
axes[1].set_title('2. Tỷ Lệ Thay Đổi Hành Động (Switching Rate %)')
axes[1].set_ylabel('Flapping Rate (%)')

for i, val in enumerate(switching_rates):
    axes[1].text(i, val + 1, f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

# --- HOÀN THIỆN XUẤT ẢNH ---
plt.tight_layout()
out_image = "results/final_comparison_bar_chart.png"
plt.savefig(out_image, dpi=300)

print(f"Đã xuất hóa ra biểu đồ cực đẹp tại: {out_image}")
plt.show() # Hiển thị ảnh cho bạn xem luôn trên màn hình
