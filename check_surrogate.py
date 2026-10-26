"""
check_surrogate.py - Kiểm tra model Digital Twin (models/surrogate_model.pkl) có phân biệt được hành động không.
Chạy trong thư mục gốc repo:  python check_surrogate.py
Không train lại, không cần testbed. Chạy mất vài giây.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, joblib
import make_figures as mf

PATH = "models/surrogate_model.pkl"
LAT_T, LOSS_T = 150.0, 0.10                        # digital_twin/safety.py
m = joblib.load(PATH)
F = mf.F

print(f"[1] Model: {m}")
leaves = [e.get_n_leaves() for e in m.estimators_]
print(f"    trees: {len(leaves)} | leaves per tree: min {min(leaves)}, max {max(leaves)}")
print(f"    feature importances: {dict(zip(F + ['action'], np.round(m.feature_importances_, 3)))}")

tr = pd.read_csv("data/processed/train_data.csv"); te = pd.read_csv("data/processed/test_data.csv")
key = lambda d: d[F].round(10).astype(str).agg("|".join, axis=1)
te = te[~key(te).isin(set(key(tr))).values].reset_index(drop=True)
raw = pd.DataFrame(mf.to_raw(te, joblib.load("models/scaler.pkl")))[F].values.astype(float)

print(f"\n[2] Clean test states ({len(raw)}), raw units, mỗi hành động a0..a4:")
for a in range(5):
    p = m.predict(np.c_[raw, np.full(len(raw), a)])
    rej = (p[:, 0] > LAT_T) | (p[:, 1] > LOSS_T)
    print(f"    a{a}: latency {p[:,0].min():.3f}..{p[:,0].max():.3f} ms | loss {p[:,1].min():.4f}..{p[:,1].max():.4f} "
          f"| reject {100*rej.mean():.1f}%")

rng = np.random.default_rng(0)
X = rng.uniform(0, 1e5, (5000, 9)); X[:, 8] = rng.integers(0, 5, 5000)
u = np.unique(m.predict(X).round(6), axis=0)
print(f"\n[3] 5,000 đầu vào ngẫu nhiên -> {len(u)} giá trị dự đoán khác nhau: {u[:3].tolist()}")
print("\nKết luận: nếu mọi cây chỉ có 1 lá và [3] chỉ có 1 giá trị, model là hằng số -> DT check không thể reject.")
