"""
make_figures.py - Đánh giá + vẽ toàn bộ hình cho paper từ model đã train (out/*.pth, out/*_curve.csv).

Chạy trong thư mục gốc repo (cần data/processed/{train,val,test}_data.csv và models/scaler.pkl
do `python -m rl_engine.data_processor` tạo ra):

    python make_figures.py                      # dùng control_loop/main_loop.py cho Rule/Guard
    python make_figures.py --main-loop <path>   # nếu main_loop.py ở chỗ khác

Kết quả trong figures/ :
  fig_training_reward.(pdf|png)   reward huấn luyện: mean ± std qua seed + từng seed mờ
  fig_reward_vs_quality.(pdf|png) mỗi seed 1 điểm: reward train cuối vs chất lượng trên tập test
  fig_per_class.(pdf|png)         tỉ lệ hành động chấp nhận được theo lớp, Rule / DQN / PPO
  fig_action_mix.(pdf|png)        phân bố hành động theo lớp (DQN vs PPO) -> thấy PPO sụp
  fig_guard.(pdf|png)             tác dụng của guard (FP và missed attack, trước -> sau)
  tables/*.csv                    mọi số liệu dùng trong hình và bảng của paper
Chạy được với số seed bất kỳ (1..n). Hình là PDF vector, rộng 1 cột IEEE (3.5 in).
"""
import argparse, ast, re
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================== cấu hình chung
F = ["packet_rate", "byte_rate", "flow_count", "flow_growth_rate",
     "src_ip_entropy", "latency", "packet_loss", "controller_cpu"]
CLASSES = ["Normal", "DDoS", "Flow\novfl.", "IP\nspoof.", "Pkt-In", "Port\nscan"]
CLS_FLAT = [c.replace("\n", " ") for c in CLASSES]
ACTIONS = ["None", "Block", "Limit", "Redirect", "Isolate"]
BEST = {0: {0}, 1: {2}, 2: {1}, 3: {1, 4}, 4: {2}, 5: {3}}        # r_sec cao nhất (offline_env.py)
OK = {0: {0}, 1: {2, 1}, 2: {1, 2}, 3: {1, 4}, 4: {2, 1}, 5: {3}}  # r_sec >= 1.5
SEEDS_RANDOM = [42, 101, 123, 456, 789]

# Màu theo THỰC THỂ, giữ nguyên trên mọi hình. Đã kiểm tra mù màu (validate_palette.js).
C = {"DQN": "#2a78d6", "PPO": "#eb6834", "Rule": "#8a8984", "Random": "#c9c8c2"}
C_ACT = ["#b8b7b1", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]   # None, Block, Limit, Redirect, Isolate
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

COL_W = 3.5  # inch, 1 cột IEEE
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print(f"  saved figures/{name}.pdf|png")


def ms(x, scale=1.0, nd=1):
    x = np.asarray(x, dtype=float) * scale
    sd = x.std(ddof=1) if len(x) > 1 else 0.0
    return f"{x.mean():.{nd}f} ± {sd:.{nd}f}"


# ============================================================== mạng (khớp dqn_agent / ppo_agent)
def _body():
    return nn.Sequential(nn.Linear(8, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())

class QNet(nn.Module):
    def __init__(self):
        super().__init__(); self.net = nn.Sequential(*_body(), nn.Linear(128, 5))
    def forward(self, x): return self.net(x)

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__(); self.shared = _body()
        self.actor = nn.Sequential(nn.Linear(128, 5), nn.Softmax(-1)); self.critic = nn.Linear(128, 1)
    def forward(self, x): return self.actor(self.shared(x))


def load_policies(out_dir):
    pols = []
    for f in sorted(out_dir.glob("*_seed*.pth")):
        m = re.match(r"(\w+)_seed(\d+)\.pth", f.name)
        if not m: continue
        algo, seed = m.group(1).upper(), int(m.group(2))
        net = QNet() if algo == "DQN" else ActorCritic()
        net.load_state_dict(torch.load(f, map_location="cpu")); net.eval()
        pols.append((algo, seed, net))
    return pols


def load_rule_funcs(main_loop):
    """Lấy NGUYÊN VĂN threat_score / is_clearly_normal / rule_action từ main_loop.py."""
    tree = ast.parse(Path(main_loop).read_text(encoding="utf-8"))
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef)
            and n.name in ("_f", "threat_score", "is_clearly_normal", "rule_action")]
    ns = {"Optional": Optional}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "main_loop_funcs", "exec"), ns)
    return ns


def to_raw(df, scaler):
    """Đảo quantile + đảo log1p của packet/byte rate (clean_data_dataset.py) -> giá trị gốc."""
    raw = pd.DataFrame(scaler.inverse_transform(df[F].values), columns=F)
    raw["packet_rate"] = np.expm1(raw["packet_rate"]); raw["byte_rate"] = np.expm1(raw["byte_rate"])
    return raw.to_dict("records")


def metrics(a, y):
    a = np.asarray(a)
    r = {"best": np.mean([ai in BEST[yi] for ai, yi in zip(a, y)]),
         "acceptable": np.mean([ai in OK[yi] for ai, yi in zip(a, y)]),
         "false_pos": np.mean(a[y == 0] != 0),
         "missed": np.mean(a[y != 0] == 0)}
    for c, n in enumerate(CLS_FLAT):
        r[f"acc_{n}"] = np.mean([ai in OK[c] for ai in a[y == c]])
    return r


def greedy_step_reward(actions, df):
    from rl_engine.offline_env import OfflineSDNEnv
    env = OfflineSDNEnv(df); env.previous_action = 0; tot = 0.0
    for i, a in enumerate(actions):
        tot += env._compute_reward(int(a), df.iloc[i]); env.previous_action = int(a)
    return tot / len(actions)


# ============================================================== đánh giá
def evaluate(args, out_dir, tab):
    te = pd.read_csv("data/processed/test_data.csv"); y = te["label"].astype(int).values
    va = pd.read_csv("data/processed/val_data.csv")
    scaler = joblib.load("models/scaler.pkl")
    fn = load_rule_funcs(args.main_loop)
    raw_te = to_raw(te, scaler)

    # guard gốc + guard hiệu chỉnh trên VALIDATION (chỉ thay ngưỡng flow_count)
    raw_va = pd.DataFrame(to_raw(va, scaler))
    flow_t = float(np.percentile(raw_va.loc[va["label"].values == 0, "flow_count"], 99))
    def normal_cal(r):
        return (r["packet_rate"] < 50 and r["flow_count"] <= flow_t and abs(r["flow_growth_rate"]) < 2
                and r["latency"] < 10 and r["packet_loss"] < 0.001 and r["controller_cpu"] < 20)
    g_orig = np.array([fn["threat_score"](r) <= 1 and fn["is_clearly_normal"](r) for r in raw_te])
    g_cal = np.array([fn["threat_score"](r) <= 1 and normal_cal(r) for r in raw_te])

    rows, mix = [], []
    X = torch.tensor(te[F].values, dtype=torch.float32)
    for algo, seed, net in load_policies(out_dir):
        with torch.no_grad(): a = net(X).argmax(1).numpy()
        rows.append({"method": algo, "seed": seed, "step_reward": greedy_step_reward(a, te), **metrics(a, y)})
        for variant, g in (("Guard(orig)", g_orig), ("Guard(cal)", g_cal)):
            rows.append({"method": f"{algo}+{variant}", "seed": seed, **metrics(np.where(g, 0, a), y)})
        for c in range(6):
            share = np.bincount(a[y == c], minlength=5) / max((y == c).sum(), 1)
            mix.append({"method": algo, "seed": seed, "cls": c, **{ACTIONS[k]: share[k] for k in range(5)}})
    ra = np.array([fn["rule_action"](r) for r in raw_te])
    rows.append({"method": "Rule", "seed": 0, "step_reward": greedy_step_reward(ra, te), **metrics(ra, y)})
    for s in SEEDS_RANDOM:
        rnd = np.random.default_rng(s).integers(0, 5, len(y))
        rows.append({"method": "Random", "seed": s, "step_reward": greedy_step_reward(rnd, te), **metrics(rnd, y)})

    ev = pd.DataFrame(rows); ev.to_csv(tab / "eval_per_seed.csv", index=False)
    mix = pd.DataFrame(mix); mix.to_csv(tab / "action_mix_per_seed.csv", index=False)

    order = [m for m in ["Random", "Rule", "DQN", "DQN+Guard(orig)", "DQN+Guard(cal)",
                         "PPO", "PPO+Guard(orig)", "PPO+Guard(cal)"] if m in set(ev.method)]
    summ = pd.DataFrame({m: {
        "n": len(g := ev[ev.method == m]),
        "best (%)": ms(g.best, 100), "acceptable (%)": ms(g.acceptable, 100),
        "false_pos (%)": ms(g.false_pos, 100), "missed (%)": ms(g.missed, 100),
        "step_reward": ms(g.step_reward, 1, 2) if g.step_reward.notna().all() else "",
        **{f"{n} (%)": ms(g[f'acc_{n}'], 100) for n in CLS_FLAT}} for m in order})
    summ.to_csv(tab / "eval_summary.csv")
    print(f"\n=== Held-out test (N={len(y)}) - mean ± std over seeds ===\n{summ.to_string()}")
    print(f"\nGuard firing on test: original {g_orig.mean()*100:.1f}% | calibrated (flow_count <= {flow_t:.0f}) "
          f"{g_cal.mean()*100:.1f}% (normal {g_cal[y==0].mean()*100:.1f}%, attack {g_cal[y!=0].mean()*100:.1f}%)")
    return ev, mix


def training_stats(out_dir, tab, w):
    curves = {}
    for f in sorted(out_dir.glob("*_seed*_curve.csv")):
        m = re.match(r"(\w+)_seed(\d+)_curve\.csv", f.name)
        curves.setdefault(m.group(1).upper(), {})[int(m.group(2))] = pd.read_csv(f)["rewards"].values
    rows = []
    for algo, d in curves.items():
        for seed, v in d.items():
            ma = pd.Series(v).rolling(w).mean().values; pos = np.where(ma > 0)[0]
            rows.append({"method": algo, "seed": seed, "episodes": len(v),
                         "final_train_reward": v[-w:].mean(),
                         "first_positive_ep": int(pos[0]) if len(pos) else np.nan,
                         "late_std": v[-200:].std()})
    tr = pd.DataFrame(rows); tr.to_csv(tab / "training_per_seed.csv", index=False)
    summ = tr.groupby("method").agg(
        n=("seed", "count"), final_train_reward=("final_train_reward", lambda x: ms(x)),
        first_positive_ep=("first_positive_ep",
                           lambda x: (ms(x.dropna(), 1, 0) if x.notna().any() else "not reached")
                           + f" ({x.notna().sum()}/{len(x)})"),
        late_std=("late_std", lambda x: ms(x)))
    summ.to_csv(tab / "training_summary.csv")
    print(f"=== Training (mean ± std over seeds) ===\n{summ.to_string()}")
    return curves, tr


# ============================================================== hình
def fig_training(curves, w, out):
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    for algo in [a for a in ("DQN", "PPO") if a in curves]:
        d = curves[algo]; L = min(len(v) for v in d.values()); x = np.arange(L)
        ma = np.stack([pd.Series(v[:L]).rolling(w).mean().values for v in d.values()])
        for row in ma:
            ax.plot(x, row, color=C[algo], lw=0.5, alpha=0.35)
        mean = np.nanmean(ma, 0)
        if len(d) > 1:
            sd = np.nanstd(ma, 0, ddof=1)
            ax.fill_between(x, mean - sd, mean + sd, color=C[algo], alpha=0.15, lw=0)
        ax.plot(x, mean, color=C[algo], lw=1.6, label=f"{algo} (mean of {len(d)} seeds)")
        ax.annotate(algo, (L - 1, mean[-1]), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=7, color=INK, fontweight="bold")
    ax.axhline(0, color=INK2, lw=0.6, ls=(0, (3, 2)))
    ax.set_xlabel("Training episode"); ax.set_ylabel(f"Episode reward (MA-{w})")
    ax.set_xlim(0, None); ax.margins(x=0.08)
    ax.legend(loc="lower right", bbox_to_anchor=(0.97, 0.2), handlelength=1.6)
    save(fig, out, "fig_training_reward")


def fig_reward_vs_quality(tr, ev, out):
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    for m, col in (("Random", C["Random"]), ("Rule", C["Rule"])):
        g = ev[ev.method == m]
        if len(g):
            yv = g.acceptable.mean() * 100
            ax.axhline(yv, color=col, lw=1.0, ls=(0, (4, 2)))
            ax.text(0.01, yv, f"{m} {yv:.1f}%", transform=ax.get_yaxis_transform(),
                    ha="left", va="bottom", fontsize=6.5, color=INK2)
    for algo, mk in (("DQN", "o"), ("PPO", "s")):
        t = tr[tr.method == algo]; e = ev[ev.method == algo]
        if t.empty or e.empty: continue
        j = t.merge(e[["seed", "acceptable"]], on="seed")
        ax.scatter(j.final_train_reward, j.acceptable * 100, s=26, marker=mk, color=C[algo],
                   edgecolor="white", lw=0.8, zorder=3, label=f"{algo} (one point per seed)")
        if algo == "DQN" and len(j) > 1 and np.ptp(j.acceptable * 100) < 2:
            ax.annotate(f"{len(j)} seeds overlap", (j.final_train_reward.mean(), j.acceptable.mean() * 100),
                        xytext=(8, -2), textcoords="offset points", fontsize=6.5, color=INK2, va="center")
    ax.set_xlabel("Final training reward (last 50 episodes)")
    ax.set_ylabel("Held-out acceptable action (%)")
    ax.set_ylim(0, 100); ax.axvline(0, color=GRID, lw=0.8, zorder=0)
    ax.legend(loc="lower right", bbox_to_anchor=(1, 0.08), handletextpad=0.3)
    save(fig, out, "fig_reward_vs_quality")


def fig_per_class(ev, out):
    methods = [m for m in ("Rule", "DQN", "PPO") if m in set(ev.method)]
    fig, ax = plt.subplots(figsize=(COL_W, 2.1))
    xs = np.arange(len(CLASSES)); w = 0.8 / len(methods); gap = 0.03
    for i, m in enumerate(methods):
        g = ev[ev.method == m]; vals = g[[f"acc_{n}" for n in CLS_FLAT]].values * 100
        mean = vals.mean(0); x = xs + (i - (len(methods) - 1) / 2) * w
        ax.bar(x, mean, w - gap, color=C[m], label=m + (f" ({len(g)} seeds)" if len(g) > 1 else ""), zorder=2)
        for k in np.where(mean < 0.5)[0]:
            ax.text(x[k], 1.5, "0", ha="center", va="bottom", fontsize=5.5, color=INK2)
        if len(g) > 1:
            jit = np.linspace(-w * 0.22, w * 0.22, len(g))
            for k in range(len(CLASSES)):
                ax.scatter(x[k] + jit, vals[:, k], s=4, color=INK, zorder=3, lw=0)
    ax.set_xticks(xs); ax.set_xticklabels(CLASSES)
    ax.set_ylim(0, 105); ax.set_ylabel("Acceptable action (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=len(methods), loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.0,
              columnspacing=1.2)
    save(fig, out, "fig_per_class")


def _stacked(ax, shares, labels):
    """shares: (n_rows, 5) in %. 100%-stacked horizontal bars with direct labels."""
    left = np.zeros(len(shares))
    for k, act in enumerate(ACTIONS):
        v = shares[:, k]
        ax.barh(range(len(shares)), v, left=left, color=C_ACT[k], height=0.72,
                edgecolor="white", linewidth=0.8, label=act, zorder=2)
        for i, (l, vi) in enumerate(zip(left, v)):
            if vi >= 18:
                ax.text(l + vi / 2, i, f"{vi:.0f}", ha="center", va="center", fontsize=6, color=INK)
        left += v
    ax.set_yticks(range(len(shares))); ax.set_yticklabels(labels); ax.invert_yaxis()
    ax.set_xlim(0, 100); ax.set_xticks([0, 50, 100]); ax.grid(False)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Share of states (%)")


def fig_action_mix(mix, out):
    """Trái: DQN theo lớp (TB qua seed, các seed nhất quán). Phải: PPO theo TỪNG SEED trên toàn tập test,
    vì lấy trung bình qua seed sẽ che mất việc mỗi seed sụp về một hành động."""
    has = set(mix.method)
    panels = (["DQN"] if "DQN" in has else []) + (["PPO"] if "PPO" in has else [])
    fig, axes = plt.subplots(1, len(panels), figsize=(COL_W, 2.0), squeeze=False,
                             gridspec_kw={"wspace": 0.75})
    for ax, algo in zip(axes[0], panels):
        g = mix[mix.method == algo]; n = g.seed.nunique()
        if algo == "DQN":
            sh = g.groupby("cls")[ACTIONS].mean().reindex(range(6)).values * 100
            _stacked(ax, sh, CLS_FLAT)
            ax.set_title(f"DQN by class (mean of {n} seed{'s' if n > 1 else ''})", fontsize=7.5, color=INK)
        else:
            per_seed = g.groupby("seed")[ACTIONS].mean().sort_index()   # lớp cân bằng -> TB theo lớp = toàn tập
            _stacked(ax, per_seed.values * 100, [f"seed {s}" for s in per_seed.index])
            ax.set_title("PPO by seed (all test states)", fontsize=7.5, color=INK)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 0.98), handlelength=1.0,
               columnspacing=0.9)
    save(fig, out, "fig_action_mix")


def fig_guard(ev, out):
    algos = [a for a in ("PPO", "DQN") if a in set(ev.method)]
    if not algos: return
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.45), sharey=True)
    for ax, (col, title) in zip(axes, (("false_pos", "False positives (%)"), ("missed", "Missed attacks (%)"))):
        for i, a in enumerate(algos):
            before = ev[ev.method == a][col].mean() * 100
            after = ev[ev.method == f"{a}+Guard(cal)"][col].mean() * 100
            ax.plot([before, after], [i, i], color=GRID, lw=2.2, zorder=1, solid_capstyle="round")
            ax.scatter([before], [i], s=30, color="white", edgecolor=C[a], lw=1.4, zorder=3)
            ax.scatter([after], [i], s=30, color=C[a], zorder=3)
            for v, ha, dx in ((before, "right" if before > after else "left", -5 if before > after else 5),
                              (after, "left" if before > after else "right", 5 if before > after else -5)):
                if abs(before - after) < 1e-9 and v is after: continue
                ax.annotate(f"{v:.0f}", (v, i), xytext=(dx, 5), textcoords="offset points",
                            ha="center", fontsize=6, color=INK2)
        ax.set_title(title, color=INK); ax.set_xlim(-8, 108); ax.set_xticks([0, 50, 100])
        ax.grid(axis="y", visible=False); ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(algos))); axes[0].set_yticklabels(algos); axes[0].set_ylim(-0.6, len(algos) - 0.4)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], marker="o", ls="", mfc="white", mec=INK2),
                Line2D([], [], marker="o", ls="", color=INK2)],
               ["without guard", "with calibrated guard"], ncol=2, loc="lower center",
               bbox_to_anchor=(0.55, 1.0))
    fig.tight_layout(w_pad=1.0)
    save(fig, out, "fig_guard")


# ============================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-loop", default="control_loop/main_loop.py")
    ap.add_argument("--out-dir", default="out", help="thư mục chứa *_seed*.pth và *_curve.csv")
    ap.add_argument("--window", type=int, default=50)
    args = ap.parse_args()
    out_dir = Path(args.out_dir); figs = Path("figures"); tab = figs / "tables"
    tab.mkdir(parents=True, exist_ok=True)

    curves, tr = training_stats(out_dir, tab, args.window)
    ev, mix = evaluate(args, out_dir, tab)
    print("\nFigures:")
    fig_training(curves, args.window, figs)
    fig_reward_vs_quality(tr, ev, figs)
    fig_per_class(ev, figs)
    fig_action_mix(mix, figs)
    fig_guard(ev, figs)


if __name__ == "__main__":
    main()
