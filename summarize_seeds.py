"""
Tổng hợp kết quả train đa seed (out/*_curve.csv, out/*.pth):
  - fig_training_mean_std.png : reward MA-50, đường trung bình ± 1 std qua các seed (Fig. 2 mới)
  - table_convergence.csv     : final reward, episode đạt reward dương, độ dao động — mean ± std
  - table_test_eval.csv       : đánh giá chính sách trên tập test giữ riêng — mean ± std
  - fig_test_per_class.png    : accuracy theo loại tấn công, cột = mean, error bar = std
"""
import glob, os, re
import numpy as np, pandas as pd, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

W = 50
F = ["packet_rate", "byte_rate", "flow_count", "flow_growth_rate",
     "src_ip_entropy", "latency", "packet_loss", "controller_cpu"]
CLS = {0: "Normal", 1: "DDoS", 2: "Flow Overflow", 3: "IP Spoofing", 4: "Packet-In", 5: "Port Scan"}
BEST = {0: {0}, 1: {2}, 2: {1}, 3: {1, 4}, 4: {2}, 5: {3}}        # r_sec cao nhất (offline_env.py)
OK = {0: {0}, 1: {2, 1}, 2: {1, 2}, 3: {1, 4}, 4: {2, 1}, 5: {3}}  # r_sec >= 1.5
COL = {"dqn": "#1f77b4", "ppo": "#ff7f0e"}

def ms(x): return f"{np.mean(x):.3f} ± {np.std(x, ddof=1) if len(x) > 1 else 0:.3f}"

# ---------- training curves ----------
curves = {}
for f in sorted(glob.glob("out/*_curve.csv")):
    algo, seed = re.match(r"(\w+)_seed(\d+)_curve.csv", os.path.basename(f)).groups()
    curves.setdefault(algo, {})[int(seed)] = pd.read_csv(f)["rewards"].values

conv_rows = []
fig, ax = plt.subplots(figsize=(6.4, 3.6))
for algo, d in curves.items():
    L = min(len(v) for v in d.values())
    ma = np.stack([pd.Series(v[:L]).rolling(W).mean().values for v in d.values()])
    m, s = np.nanmean(ma, 0), np.nanstd(ma, 0, ddof=1) if len(d) > 1 else np.zeros(L)
    x = np.arange(L)
    ax.plot(x, m, color=COL[algo], label=f"{algo.upper()} (n={len(d)} seeds)")
    ax.fill_between(x, m - s, m + s, color=COL[algo], alpha=0.25)
    for seed, v in d.items():
        mav = pd.Series(v).rolling(W).mean().values
        pos = np.where(mav > 0)[0]
        conv_rows.append({"algo": algo.upper(), "seed": seed, "episodes": len(v),
                          "final_reward_last50": v[-W:].mean(),
                          "episode_first_positive_MA50": int(pos[0]) if len(pos) else np.nan,
                          "reward_std_last200": v[-200:].std()})
ax.axhline(0, color="gray", lw=0.6, ls="--")
ax.set_xlabel("Episode"); ax.set_ylabel(f"Episode reward (MA-{W})")
ax.legend(); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig("fig_training_mean_std.png", dpi=300)

conv = pd.DataFrame(conv_rows)
conv.to_csv("table_convergence_per_seed.csv", index=False)
agg = conv.groupby("algo").agg(
    n_seeds=("seed", "count"),
    final_reward=("final_reward_last50", ms),
    first_positive_ep=("episode_first_positive_MA50", lambda x: ms(x.dropna()) + f" ({x.notna().sum()}/{len(x)} reached)"),
    late_reward_std=("reward_std_last200", ms))
agg.to_csv("table_convergence.csv"); print("=== Convergence (mean ± std over seeds) ===\n", agg.to_string())

# ---------- held-out test evaluation ----------
def mlp(): return nn.Sequential(nn.Linear(8, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
class Q(nn.Module):
    def __init__(s): super().__init__(); s.net = nn.Sequential(*mlp(), nn.Linear(128, 5))
    def forward(s, x): return s.net(x)
class AC(nn.Module):
    def __init__(s):
        super().__init__(); s.shared = mlp()
        s.actor = nn.Sequential(nn.Linear(128, 5), nn.Softmax(-1)); s.critic = nn.Linear(128, 1)
    def forward(s, x): return s.actor(s.shared(x))

te = pd.read_csv("data/processed/test_data.csv")
X = torch.tensor(te[F].values, dtype=torch.float32); y = te["label"].astype(int).values
rows = []
for f in sorted(glob.glob("out/*_seed*.pth")):
    algo, seed = re.match(r"(\w+)_seed(\d+).pth", os.path.basename(f)).groups()
    net = Q() if algo == "dqn" else AC(); net.load_state_dict(torch.load(f, map_location="cpu")); net.eval()
    with torch.no_grad(): a = net(X).argmax(1).numpy()
    r = {"algo": algo.upper(), "seed": int(seed),
         "best_action_acc": np.mean([ai in BEST[yi] for ai, yi in zip(a, y)]),
         "acceptable_action_acc": np.mean([ai in OK[yi] for ai, yi in zip(a, y)]),
         "false_positive_rate": np.mean(a[y == 0] != 0),
         "missed_attack_rate": np.mean(a[y != 0] == 0)}
    for c, n in CLS.items(): r[f"acc_{n}"] = np.mean([ai in OK[c] for ai in a[y == c]])
    rows.append(r)
ev = pd.DataFrame(rows); ev.to_csv("table_test_eval_per_seed.csv", index=False)
metric_cols = [c for c in ev.columns if c not in ("algo", "seed")]
evagg = ev.groupby("algo")[metric_cols].agg(ms); evagg.insert(0, "n_seeds", ev.groupby("algo").size())
evagg.T.to_csv("table_test_eval.csv"); print(f"\n=== Held-out test (N={len(y)}), mean ± std over seeds ===\n", evagg.T.to_string())

fig, ax = plt.subplots(figsize=(6.4, 3.4)); w = 0.38; xs = np.arange(len(CLS))
for i, algo in enumerate(sorted(ev.algo.unique())):
    sub = ev[ev.algo == algo]; cols = [f"acc_{n}" for n in CLS.values()]
    ax.bar(xs + (i - 0.5) * w, sub[cols].mean(), w, yerr=sub[cols].std(ddof=1), capsize=3,
           color=COL[algo.lower()], label=f"{algo} (n={len(sub)})")
ax.set_xticks(xs); ax.set_xticklabels(CLS.values(), rotation=20); ax.set_ylim(0, 1.05)
ax.set_ylabel("Correct-action rate"); ax.legend(); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig("fig_test_per_class.png", dpi=300)
print("\nSaved: fig_training_mean_std.png, fig_test_per_class.png, table_convergence.csv, table_test_eval.csv")
