"""
B.2: baseline Rule + Random trên cùng tập test.
B.3: tác dụng của guard (RL có/không có guard).
Hàm threat_score / is_clearly_normal / rule_action lấy NGUYÊN VĂN từ control_loop/main_loop.py (qua ast).
Cách dùng: python3 eval_baselines.py <đường dẫn main_loop.py>
"""
import ast, glob, os, re, sys
import joblib, numpy as np, pandas as pd, torch, torch.nn as nn

F = ["packet_rate", "byte_rate", "flow_count", "flow_growth_rate",
     "src_ip_entropy", "latency", "packet_loss", "controller_cpu"]
CLS = {0: "Normal", 1: "DDoS", 2: "Flow Overflow", 3: "IP Spoofing", 4: "Packet-In", 5: "Port Scan"}
BEST = {0: {0}, 1: {2}, 2: {1}, 3: {1, 4}, 4: {2}, 5: {3}}
OK = {0: {0}, 1: {2, 1}, 2: {1, 2}, 3: {1, 4}, 4: {2, 1}, 5: {3}}

# ---- lấy hàm gốc từ main_loop.py ----
src = open(sys.argv[1], encoding="utf-8").read()
tree = ast.parse(src)
keep = [n for n in tree.body if isinstance(n, ast.FunctionDef)
        and n.name in ("_f", "threat_score", "is_clearly_normal", "rule_action")]
from typing import Optional
ns = {"Optional": Optional}
exec(compile(ast.Module(body=keep, type_ignores=[]), "main_loop_funcs", "exec"), ns)
rule_action, threat_score, is_clearly_normal = ns["rule_action"], ns["threat_score"], ns["is_clearly_normal"]

# ---- tập test: scaled -> giá trị gốc (đảo quantile, đảo log1p của packet/byte rate) ----
te = pd.read_csv("data/processed/test_data.csv")
y = te["label"].astype(int).values
scaler = joblib.load("models/scaler.pkl")
raw = pd.DataFrame(scaler.inverse_transform(te[F].values), columns=F)
raw["packet_rate"] = np.expm1(raw["packet_rate"]); raw["byte_rate"] = np.expm1(raw["byte_rate"])
raw_rows = raw.to_dict("records")

def metrics(a):
    a = np.asarray(a)
    r = {"best_action_acc": np.mean([ai in BEST[yi] for ai, yi in zip(a, y)]),
         "acceptable_action_acc": np.mean([ai in OK[yi] for ai, yi in zip(a, y)]),
         "false_positive_rate": np.mean(a[y == 0] != 0),
         "missed_attack_rate": np.mean(a[y != 0] == 0)}
    for c, n in CLS.items(): r[f"acc_{n}"] = np.mean([ai in OK[c] for ai in a[y == c]])
    return r

rows = []
# Rule (tất định)
rows.append({"method": "Rule", "seed": 0, **metrics([rule_action(r) for r in raw_rows])})
# Random (5 seed)
for s in [42, 101, 123, 456, 789]:
    rows.append({"method": "Random", "seed": s, **metrics(np.random.default_rng(s).integers(0, 5, len(y)))})

# ---- RL có / không có guard ----
def mlp(): return nn.Sequential(nn.Linear(8, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
class Q(nn.Module):
    def __init__(s): super().__init__(); s.net = nn.Sequential(*mlp(), nn.Linear(128, 5))
    def forward(s, x): return s.net(x)
class AC(nn.Module):
    def __init__(s):
        super().__init__(); s.shared = mlp()
        s.actor = nn.Sequential(nn.Linear(128, 5), nn.Softmax(-1)); s.critic = nn.Linear(128, 1)
    def forward(s, x): return s.actor(s.shared(x))

guard_mask = np.array([threat_score(r) <= 1.0 and is_clearly_normal(r) for r in raw_rows])

# Guard hiệu chỉnh: CHỈ thay ngưỡng flow_count (gốc < 15) bằng phân vị 99 của flow_count
# ở lớp Normal trên tập VALIDATION. Các ngưỡng khác giữ nguyên như is_clearly_normal gốc.
va = pd.read_csv("data/processed/val_data.csv")
va_raw = pd.DataFrame(scaler.inverse_transform(va[F].values), columns=F)
FLOW_T = float(np.percentile(va_raw.loc[va["label"].values == 0, "flow_count"], 99))
def clearly_normal_cal(r):
    return (r["packet_rate"] < 50 and r["flow_count"] <= FLOW_T and abs(r["flow_growth_rate"]) < 2
            and r["latency"] < 10 and r["packet_loss"] < 0.001 and r["controller_cpu"] < 20)
guard_cal = np.array([threat_score(r) <= 1.0 and clearly_normal_cal(r) for r in raw_rows])
print(f"Calibrated flow_count threshold (val, normal, p99): <= {FLOW_T:.1f}")
X = torch.tensor(te[F].values, dtype=torch.float32)
for f in sorted(glob.glob("out/*_seed*.pth")):
    algo, seed = re.match(r"(\w+)_seed(\d+).pth", os.path.basename(f)).groups()
    net = Q() if algo == "dqn" else AC(); net.load_state_dict(torch.load(f, map_location="cpu")); net.eval()
    with torch.no_grad(): a = net(X).argmax(1).numpy()
    rows.append({"method": algo.upper(), "seed": int(seed), **metrics(a)})
    rows.append({"method": algo.upper() + "+Guard", "seed": int(seed), **metrics(np.where(guard_mask, 0, a))})
    rows.append({"method": algo.upper() + "+GuardCal", "seed": int(seed), **metrics(np.where(guard_cal, 0, a))})

TABLES_DIR = os.path.join("figures", "tables")
os.makedirs(TABLES_DIR, exist_ok=True)
df = pd.DataFrame(rows); df.to_csv(os.path.join(TABLES_DIR, "table_all_methods_per_seed.csv"), index=False)
cols = [c for c in df.columns if c not in ("method", "seed")]
def ms(x): return f"{x.mean()*100:.1f} ± {x.std(ddof=1)*100 if len(x) > 1 else 0:.1f}"
order = [m for m in ["Random", "Rule", "DQN", "DQN+Guard", "DQN+GuardCal", "PPO", "PPO+Guard", "PPO+GuardCal"]
         if m in df.method.unique()]
agg = df.groupby("method")[cols].agg(ms).loc[order]
agg.insert(0, "n", df.groupby("method").size().loc[order])
agg.T.to_csv(os.path.join(TABLES_DIR, "table_all_methods.csv"))
pd.set_option("display.width", 250)
print(f"=== Held-out test (N={len(y)}), % , mean ± std over seeds ===\n", agg.T.to_string())
print(f"\nGuard fires on {guard_mask.mean()*100:.1f}% of test states "
      f"(normal: {guard_mask[y==0].mean()*100:.1f}%, attack: {guard_mask[y!=0].mean()*100:.1f}%)")
print(f"Calibrated guard fires on {guard_cal.mean()*100:.1f}% of test states "
      f"(normal: {guard_cal[y==0].mean()*100:.1f}%, attack: {guard_cal[y!=0].mean()*100:.1f}%)")
for c, n in CLS.items(): print(f"   fires on {n:13s}: {guard_cal[y==c].mean()*100:5.1f}%")
