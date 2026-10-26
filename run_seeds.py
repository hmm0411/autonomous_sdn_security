"""Train lại offline đa seed, lưu reward/loss từng episode và model từng seed (không dùng MLflow)."""
import os, sys, time, json, torch, pandas as pd
torch.set_num_threads(1)
algo = sys.argv[1]
os.makedirs("out", exist_ok=True)
df = pd.read_csv("data/processed/train_data.csv")
if algo == "ppo":
    from rl_engine.agent import train_ppo as m; run = m.run_single_seed_ppo
else:
    from rl_engine.agent import train_dqn as m; run = m.run_single_seed_dqn
for seed in [int(s) for s in sys.argv[2].split(",")]:
    t = time.time()
    res, agent = run(seed, df)
    pd.DataFrame({k: v for k, v in res.items() if isinstance(v, list)}).to_csv(f"out/{algo}_seed{seed}_curve.csv", index_label="episode")
    net = agent.model if algo == "ppo" else agent.q_net
    torch.save(net.state_dict(), f"out/{algo}_seed{seed}.pth")
    print(json.dumps({"algo": algo, "seed": seed, "sec": round(time.time()-t), "final50": float(pd.Series(res["rewards"][-50:]).mean())}), flush=True)
