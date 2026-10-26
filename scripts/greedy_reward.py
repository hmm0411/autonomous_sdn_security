"""Reward trung bình mỗi bước của chính sách GREEDY trên tập test, tính bằng đúng hàm reward huấn luyện
(OfflineSDNEnv._compute_reward), duyệt tuần tự toàn bộ tập test (có tính switching penalty)."""
import glob, os, re, sys, ast, joblib, numpy as np, pandas as pd, torch, torch.nn as nn
from typing import Optional
from rl_engine.offline_env import OfflineSDNEnv
F=["packet_rate","byte_rate","flow_count","flow_growth_rate","src_ip_entropy","latency","packet_loss","controller_cpu"]
te=pd.read_csv("data/processed/test_data.csv")
def mlp(): return nn.Sequential(nn.Linear(8,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU())
class Q(nn.Module):
    def __init__(s): super().__init__(); s.net=nn.Sequential(*mlp(),nn.Linear(128,5))
    def forward(s,x): return s.net(x)
class AC(nn.Module):
    def __init__(s): super().__init__(); s.shared=mlp(); s.actor=nn.Sequential(nn.Linear(128,5),nn.Softmax(-1)); s.critic=nn.Linear(128,1)
    def forward(s,x): return s.actor(s.shared(x))
def mean_reward(actions):
    env=OfflineSDNEnv(te); env.previous_action=0; tot=0.0
    for i,a in enumerate(actions):
        tot+=env._compute_reward(int(a),te.iloc[i]); env.previous_action=int(a)
    return tot/len(actions)
X=torch.tensor(te[F].values,dtype=torch.float32); rows=[]
for f in sorted(glob.glob("out/*_seed*.pth")):
    algo,seed=re.match(r"(\w+)_seed(\d+).pth",os.path.basename(f)).groups()
    net=Q() if algo=="dqn" else AC(); net.load_state_dict(torch.load(f,map_location="cpu")); net.eval()
    with torch.no_grad(): a=net(X).argmax(1).numpy()
    dist=np.bincount(a,minlength=5)/len(a)
    rows.append({"method":algo.upper(),"seed":int(seed),"mean_step_reward":mean_reward(a),**{f"share_a{k}":round(dist[k],3) for k in range(5)}})
# Random + Rule
src=open(sys.argv[1]).read(); tree=ast.parse(src); ns={"Optional":Optional}
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in("_f","threat_score","is_clearly_normal","rule_action")],type_ignores=[]),"m","exec"),ns)
sc=joblib.load("models/scaler.pkl"); raw=pd.DataFrame(sc.inverse_transform(te[F].values),columns=F)
raw["packet_rate"]=np.expm1(raw.packet_rate); raw["byte_rate"]=np.expm1(raw.byte_rate)
ra=np.array([ns["rule_action"](r) for r in raw.to_dict("records")])
rows.append({"method":"Rule","seed":0,"mean_step_reward":mean_reward(ra),**{f"share_a{k}":round(np.mean(ra==k),3) for k in range(5)}})
for s in [42,101,123,456,789]:
    a=np.random.default_rng(s).integers(0,5,len(te))
    rows.append({"method":"Random","seed":s,"mean_step_reward":mean_reward(a),**{f"share_a{k}":round(np.mean(a==k),3) for k in range(5)}})
os.makedirs(os.path.join("figures", "tables"), exist_ok=True)
df=pd.DataFrame(rows); df.to_csv(os.path.join("figures", "tables", "table_greedy_reward_per_seed.csv"),index=False)
print(df.to_string(index=False))
g=df.groupby("method").mean_step_reward.agg(["mean","std","count"]); g.to_csv(os.path.join("figures", "tables", "table_greedy_reward.csv")); print(g.round(3))
