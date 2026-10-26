"""
classifier_baseline.py - Baseline "classifier rồi chọn hành động tốt nhất cho lớp dự đoán".
Trả lời câu hỏi reviewer: "sao không dùng classifier thay cho RL?"

- Train trên data/processed/train_data.csv (cùng 8 đặc trưng, cùng tập train với DQN/PPO), 5 seed.
- Hành động = argmax r_sec của lớp dự đoán (Bảng I).
- Chấm trên tập test sạch (2.231 mẫu) bằng đúng chỉ số trong paper: acceptable, FP, missed, reward/bước
  (hàm reward offline_env, có clip [-5, 3]); so với DQN 5 seed trong out/.
Chạy trong thư mục gốc repo:  python classifier_baseline.py      (khoảng 1-3 phút, CPU)
Kết quả: figures/tables/classifier_baseline.csv
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score
import make_figures as mf

F = mf.F; SEEDS = [42, 101, 123, 456, 789]
tr = pd.read_csv("data/processed/train_data.csv"); te = pd.read_csv("data/processed/test_data.csv")
key = lambda d: d[F].round(10).astype(str).agg("|".join, axis=1)
te = te[~key(te).isin(set(key(tr))).values].reset_index(drop=True)
Xtr, ytr = tr[F].values.astype(float), tr.label.values
X, y = te[F].values.astype(float), te.label.values
print(f"train {len(ytr)} | clean test {len(y)}")

# ---- reward của offline_env (giống reward_sensitivity.py, tham số mặc định)
RSEC = {0: [2.5, -3.5, -2.0, -2.5, -4.5], 1: [-3.0, 0.8, 2.5, 0.4, -0.8], 2: [-3.0, 2.5, 2.0, -0.5, 0.5],
        3: [-3.2, 2.2, -0.3, 0.8, 1.2], 4: [-3.2, 1.7, 2.6, -0.3, 0.3], 5: [-2.8, 0.8, 0.3, 2.8, -0.5]}
KAPPA = np.array([0.0, 0.12, 0.08, 0.10, 0.75])
sc = lambda v: np.clip(np.abs(v), 0, 1)
pk, by, fl, gr, en, la, lo, cp = (sc(X[:, i]) for i in range(8)); lo = np.clip(X[:, 6], 0, 1)
QOS = 0.10*la + 0.20*lo + 0.10*cp + 0.10*gr
RHO = 0.20*pk + 0.15*by + 0.15*fl + 0.20*gr + 0.15*en + 0.15*cp

def evaluate(a):
    a = np.asarray(a, int); prev = np.concatenate([[0], a[:-1]])
    sec = np.array([2.8 if (yi == 3 and ai == 4 and rh >= 0.55) else RSEC[yi][ai] for yi, ai, rh in zip(y, a, RHO)])
    att, act = y != 0, a != 0
    r = np.clip(sec + 0.4*RHO*(att & act) - QOS - KAPPA[a] - 0.03*(a != prev) - 1.2*((~att) & act), -5, 3)
    ok = np.array([ai in mf.OK[yi] for yi, ai in zip(y, a)])
    return {"acceptable (%)": 100*ok.mean(), "FP (%)": 100*act[~att].mean(),
            "missed (%)": 100*(~act)[att].mean(), "reward/step": r.mean()}

BEST_ACT = {c: int(np.argmax(RSEC[c])) for c in RSEC}          # hành động r_sec cao nhất mỗi lớp
models = {"RF classifier": lambda s: RandomForestClassifier(n_estimators=300, random_state=s, n_jobs=-1),
          "MLP classifier (2x128)": lambda s: MLPClassifier((128, 128), max_iter=300, random_state=s)}
rows = []
for name, make in models.items():
    for s in SEEDS:
        clf = make(s).fit(Xtr, ytr); pred = clf.predict(X)
        r = {"policy": name, "seed": s, "balanced acc. (class)": balanced_accuracy_score(y, pred)}
        r.update(evaluate([BEST_ACT[c] for c in pred])); rows.append(r)
        print(f"{name:24s} seed {s}: " + " | ".join(f"{k} {v:.3f}" for k, v in r.items() if k not in ("policy", "seed")), flush=True)

for algo, s, net in mf.load_policies(Path("out")):
    if algo != "DQN": continue
    with torch.no_grad(): a = net(torch.tensor(X, dtype=torch.float32)).argmax(1).numpy()
    rows.append({"policy": "DQN", "seed": s, **evaluate(a)})
rows.append({"policy": "Oracle", "seed": "-", **evaluate([BEST_ACT[c] for c in y])})

df = pd.DataFrame(rows); Path("figures/tables").mkdir(parents=True, exist_ok=True)
df.to_csv("figures/tables/classifier_baseline.csv", index=False)
cols = ["acceptable (%)", "FP (%)", "missed (%)", "reward/step", "balanced acc. (class)"]
summ = df.groupby("policy", sort=False)[cols].agg(["mean", "std"]).round(3)
pd.set_option("display.width", 250); print("\n", summ.to_string())
print("\nPer-class acceptable (%), seed 42:")
for name in list(models) + ["DQN"]:
    if name == "DQN":
        net = [n for al, s, n in mf.load_policies(Path("out")) if al == "DQN" and s == 42][0]
        with torch.no_grad(): a = net(torch.tensor(X, dtype=torch.float32)).argmax(1).numpy()
    else:
        a = np.array([BEST_ACT[c] for c in models[name](42).fit(Xtr, ytr).predict(X)])
    pc = {mf.CLS_FLAT[c]: round(100*np.mean([ai in mf.OK[c] for ai in a[y == c]]), 1) for c in np.unique(y)}
    print(f"  {name:24s} {pc}")
