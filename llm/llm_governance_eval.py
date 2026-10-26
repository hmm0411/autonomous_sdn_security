"""
llm_governance_eval.py - Đánh giá lớp giải thích LLM trên quyết định THẬT của DQN/PPO (tập test sạch).

Prompt: đúng build_explanation_prompt() của repo, attack_context=None (LLM KHÔNG thấy nhãn tấn công).
Chỉ số mỗi bản ghi:
  format_ok           đủ 4 mục WHY / QoS IMPACT / SAFETY VERDICT / SUMMARY
  action_consistent   lời giải thích nêu đúng hành động đã chọn
  faithfulness        tỉ lệ con số trong lời giải thích khớp giá trị đầu vào (giống compute_faithfulness_rule)
  verdict             SAFE / RISKY / UNPARSED
  decision_ok         hành động có "acceptable" theo r_sec (ground truth, LLM không thấy)
Governance: RISKY-recall trên quyết định SAI, false-alarm (RISKY) trên quyết định ĐÚNG.

Chạy (trong thư mục gốc repo, cần models/scaler.pkl, data/processed/*_data.csv, out/*.pth):
  python llm_governance_eval.py --dry-run                 # kiểm tra, không gọi API
  $env:HF_TOKEN="hf_..."; python llm_governance_eval.py   # chạy thật (mặc định 20 mẫu/lớp/model)
Kết quả: figures/tables/llm_governance_records.csv, llm_governance_summary.csv
"""
import argparse, re, time
from pathlib import Path
import numpy as np, pandas as pd, torch, joblib
from scripts import make_figures as mf
from llm.prompt_builder import build_explanation_prompt, ACTION_MAP

ap = argparse.ArgumentParser()
ap.add_argument("--per-class", type=int, default=20, help="số trạng thái mỗi lớp cho mỗi model")
ap.add_argument("--seeds", default="42", help="seed của model DQN/PPO dùng để sinh quyết định")
ap.add_argument("--sleep", type=float, default=0.5, help="giây nghỉ giữa các lần gọi API")
ap.add_argument("--max-tokens", type=int, default=400)
ap.add_argument("--temperature", type=float, default=0.1)
ap.add_argument("--dry-run", action="store_true", help="không gọi API, dùng câu trả lời giả để kiểm tra pipeline")
args = ap.parse_args()

F = mf.F
tab = Path("figures/tables"); tab.mkdir(parents=True, exist_ok=True)
tr = pd.read_csv("data/processed/train_data.csv"); te = pd.read_csv("data/processed/test_data.csv")
key = lambda d: d[F].round(10).astype(str).agg("|".join, axis=1)
te = te[~key(te).isin(set(key(tr))).values].reset_index(drop=True)
raw = pd.DataFrame(mf.to_raw(te, joblib.load("models/scaler.pkl"))).round(3)
y = te.label.values
rng = np.random.default_rng(2026)
idx = np.concatenate([rng.choice(np.where(y == c)[0], min(args.per_class, (y == c).sum()), replace=False)
                      for c in np.unique(y)])

KEYWORDS = {0: ["no_action", "no action", "not to act", "no mitigation"],
            1: ["block_suspicious_flow", "block"], 2: ["limit_bandwidth", "limit", "rate-limit", "rate limit"],
            3: ["redirect_traffic", "redirect", "honeypot"], 4: ["isolate_device", "isolat"]}

def faithfulness(text, state):
    valid = {round(float(v), 2) for v in state.values()}
    nums = [round(float(x), 2) for x in re.findall(r"\b\d+(?:\.\d+)?\b", text)]
    return 1.0 if not nums else sum(n in valid for n in nums) / len(nums)

def verdict(text):
    m = re.search(r"SAFETY VERDICT\s*:?\s*\**\s*(SAFE|RISKY)", text, re.I)
    return m.group(1).upper() if m else "UNPARSED"

if args.dry_run:
    def call_llm(prompt, max_tokens, temperature):
        act = re.search(r'Action selected: (\w+)', prompt).group(1)
        return {"text": f"WHY: high packet_rate.\nQoS IMPACT: latency 5.0 ms.\nSAFETY VERDICT: SAFE — ok.\n"
                        f"SUMMARY: The agent chose {act}.", "latency_ms": 0.0}
else:
    import os
    from llm.llm_service import call_llm, HF_MODEL
    print(f"HF_MODEL = {HF_MODEL} | HF_TOKEN set: {bool(os.getenv('HF_TOKEN'))}", flush=True)
    probe = call_llm("Reply with the single word OK.", max_tokens=5, temperature=0.0)
    if probe["text"].startswith("LLM_ERROR"):
        print("\nAPI CHECK FAILED - no evaluation was run. Full error message:\n")
        print(probe["text"])
        raise SystemExit(1)
    print(f"API check OK: {probe['text']!r} ({probe['latency_ms']:.0f} ms)\n", flush=True)

pols = {(a, s): n for a, s, n in mf.load_policies(Path("out")) if str(s) in args.seeds.split(",")}
rows = []
for (algo, seed), net in sorted(pols.items()):
    with torch.no_grad():
        acts = net(torch.tensor(te.loc[idx, F].values, dtype=torch.float32)).argmax(1).numpy()
    for i, a in zip(idx, acts):
        state = {k: float(raw.loc[i, k]) for k in F}
        qos = {"latency": state["latency"], "packet_loss": state["packet_loss"], "throughput": state["byte_rate"]}
        prompt = build_explanation_prompt(state, int(a), qos, attack_context=None)
        out = call_llm(prompt, max_tokens=args.max_tokens, temperature=args.temperature)
        text = out["text"]; low = text.lower()
        rows.append({"model": algo, "seed": seed, "test_index": int(i), "true_class": mf.CLS_FLAT[y[i]],
                     "action": ACTION_MAP[int(a)], "decision_ok": int(int(a) in mf.OK[y[i]]),
                     "llm_error": int(text.startswith("LLM_ERROR")),
                     "format_ok": int(all(h in text.upper() for h in ("WHY", "QOS IMPACT", "SAFETY VERDICT", "SUMMARY"))),
                     "action_consistent": int(any(k in low for k in KEYWORDS[int(a)])),
                     "faithfulness": faithfulness(text, state), "verdict": verdict(text),
                     "latency_s": out["latency_ms"] / 1000.0, "text": text})
        print(f"{algo} {mf.CLS_FLAT[y[i]]:13s} {ACTION_MAP[int(a)]:22s} ok={rows[-1]['decision_ok']} "
              f"verdict={rows[-1]['verdict']} faith={rows[-1]['faithfulness']:.2f}", flush=True)
        if rows[-1]["llm_error"]:
            print("   LLM_ERROR:", text[:300], flush=True)
            if sum(r["llm_error"] for r in rows[-5:]) == 5:
                print("\n5 consecutive API errors - stopping. Fix the error above and rerun.")
                pd.DataFrame(rows).to_csv(tab / "llm_governance_records.csv", index=False)
                raise SystemExit(1)
        if not args.dry_run: time.sleep(args.sleep)

df = pd.DataFrame(rows); df.to_csv(tab / "llm_governance_records.csv", index=False)
ok = df[df.llm_error == 0]
if ok.empty:
    print(f"All {len(df)} calls failed; no summary. First error:\n{df.text.iloc[0]}"); raise SystemExit(1)

def summ(g):
    wrong, right = g[g.decision_ok == 0], g[g.decision_ok == 1]
    return pd.Series({
        "records": len(g),
        "format_ok (%)": 100 * g.format_ok.mean(),
        "action_consistent (%)": 100 * g.action_consistent.mean(),
        "faithfulness (mean)": g.faithfulness.mean(),
        "verdict parsed (%)": 100 * (g.verdict != "UNPARSED").mean(),
        "wrong decisions": len(wrong),
        "RISKY on wrong decisions (%)": 100 * (wrong.verdict == "RISKY").mean() if len(wrong) else np.nan,
        "RISKY on correct decisions (%)": 100 * (right.verdict == "RISKY").mean() if len(right) else np.nan,
        "median latency (s)": g.latency_s.median()})

s = ok.groupby("model").apply(summ).T
s["All"] = summ(ok)
s.to_csv(tab / "llm_governance_summary.csv")
pd.set_option("display.width", 200)
print(f"\nAPI errors: {int(df.llm_error.sum())} of {len(df)}\n")
print(s.round(3).to_string())
