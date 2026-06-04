"""
llm_evaluator.py  (v2 — dữ liệu thật từ RL training)
------------------------------------------------------
Đánh giá LLM Cognition Layer bằng dữ liệu THẬT lấy từ:
  • TensorBoard runs/   → training summary (reward, loss, epsilon)
  • Model replay        → per-step (state, action, reward) của DQN/PPO đã train

Ba kịch bản đánh giá lấy từ dữ liệu thật:
  1. Scenario Normal  → các bước model chọn trong điều kiện bình thường (attack_indicator ≤ 0.2)
  2. Scenario Attack  → các bước model phải phòng vệ (attack_indicator > 0.5)
  3. Scenario Noisy   → state bị xóa 3 features quan trọng → test faithfulness

Metrics đo:
  Định lượng: Strict Accuracy, Faithfulness (rule-based), Latency, Answer Relevancy
  Định tính : Accuracy/Clarity/Usefulness (LLM-as-Judge 1–5)

Usage:
  python -m llm.llm_evaluator
  python -m llm.llm_evaluator --model ppo --n-attack 10 --no-llm-judge
  python -m llm.llm_evaluator --model dqn --output results/eval.csv
"""
import os, sys, re, json, time, csv, argparse
from datetime import datetime
from typing import Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm.llm_service   import call_llm, call_llm_json
from llm.prompt_builder import (
    build_explanation_prompt,
    build_faithfulness_probe_prompt,
    build_judge_prompt,
    ACTION_MAP, VALID_ACTIONS,
)
from llm.data_loader import (
    sample_decisions,
    get_training_summary,
    export_replay_to_csv,
)

DEFENSE_ACTIONS = {1, 2, 3, 4}

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ═══════════════════════════════════════════════════════════════════════════════
# METRIC FUNCTIONS  (giống v1, không đổi logic)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_strict_accuracy(llm_text: str, expected_name: str,
                            accepted_set: Optional[set] = None) -> dict:
    text_lower = llm_text.lower()
    found = [a for a in VALID_ACTIONS if a in text_lower]
    if len(found) == 1:
        predicted = found[0]
    elif len(found) > 1:
        predicted = min((text_lower.find(a), a) for a in found)[1]
    else:
        predicted = "UNKNOWN"

    exact  = (predicted == expected_name)
    in_set = bool(accepted_set) and (predicted in {ACTION_MAP[i] for i in accepted_set})
    return {"predicted_action": predicted, "exact_match": exact,
            "in_accepted_set": in_set if accepted_set else exact}


def compute_faithfulness_rule(llm_text: str, source_state: dict) -> dict:
    valid = set()
    for v in source_state.values():
        try:
            valid.add(round(float(v), 2))
        except (TypeError, ValueError):
            pass
    extracted = [round(float(x), 2) for x in re.findall(r"\b\d+(?:\.\d+)?\b", llm_text)]
    if not extracted:
        return {"score": 1.0, "hallucinated": [], "faithful": [], "note": "no_numbers"}
    faithful     = [x for x in extracted if x in valid]
    hallucinated = [x for x in extracted if x not in valid]
    return {
        "score":        round(len(faithful) / len(extracted), 3),
        "hallucinated": list(set(hallucinated)),
        "faithful":     list(set(faithful)),
    }


def compute_faithfulness_llm(llm_text: str, source_state: dict) -> dict:
    probe  = build_faithfulness_probe_prompt(llm_text, source_state)
    result = call_llm_json(probe, max_tokens=300)
    if result.get("parse_error"):
        return {"score": None, "error": "parse_fail", "judge_latency_ms": result.get("latency_ms")}
    return {
        "score":          result.get("faithfulness_score"),
        "verdict":        result.get("verdict"),
        "hallucinations": result.get("hallucinations", []),
        "judge_latency_ms": result.get("latency_ms"),
    }


def compute_qualitative(llm_text: str, action_name: str, state: dict) -> dict:
    prompt = build_judge_prompt(llm_text, action_name, state)
    result = call_llm_json(prompt, max_tokens=200)
    if result.get("parse_error"):
        return {"error": "parse_fail", "judge_latency_ms": result.get("latency_ms")}
    return {
        "qual_accuracy":   result.get("accuracy"),
        "qual_clarity":    result.get("clarity"),
        "qual_usefulness": result.get("usefulness"),
        "justification":   result.get("justification", "")[:200],
        "judge_latency_ms": result.get("latency_ms"),
    }


def compute_answer_relevancy(text: str) -> float:
    kws = ["sdn","network","traffic","attack","block","flow","packet",
           "latency","bandwidth","anomaly","defense","isolate","redirect",
           "security","controller"]
    tl = text.lower()
    return round(sum(1 for k in kws if k in tl) / len(kws), 3)


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATE ONE RECORD
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_record(rec: dict, scenario: str, use_llm_judge: bool) -> dict:
    """Đánh giá một record (state, action, reward, qos) từ model replay."""
    state_dict  = rec["state_dict"]
    action_id   = rec["action_id"]
    action_name = rec["action_name"]
    qos         = rec["qos"]
    attack_ctx  = rec["attack_label"]

    # ── Sinh explanation ──────────────────────────────────────────────────────
    prompt     = build_explanation_prompt(state_dict, action_id, qos, attack_ctx)
    llm_result = call_llm(prompt, max_tokens=350)
    llm_text   = llm_result["text"]
    latency_ms = llm_result["latency_ms"]

    # ── Strict Accuracy ───────────────────────────────────────────────────────
    # Với attack scenario: chấp nhận bất kỳ hành động phòng vệ nào là đúng
    if scenario == "attack" and action_id in DEFENSE_ACTIONS:
        acc = compute_strict_accuracy(llm_text, action_name, DEFENSE_ACTIONS)
    else:
        acc = compute_strict_accuracy(llm_text, action_name)

    # ── Faithfulness ─────────────────────────────────────────────────────────
    faith_rule = compute_faithfulness_rule(llm_text, state_dict)
    faith_llm  = compute_faithfulness_llm(llm_text, state_dict) if use_llm_judge else {}

    # ── Qualitative ───────────────────────────────────────────────────────────
    qual = compute_qualitative(llm_text, action_name, state_dict) if use_llm_judge else {}

    # ── Answer Relevancy ──────────────────────────────────────────────────────
    relevancy = compute_answer_relevancy(llm_text)

    return {
        "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scenario":         scenario,
        "model":            rec["model"],
        "step":             rec["step"],
        # Ground truth từ RL
        "rl_action":        action_name,
        "rl_reward":        rec["reward"],
        "attack_indicator": rec["attack_indicator"],
        "attack_label":     attack_ctx,
        # LLM response (truncated)
        "llm_response":     llm_text[:400].replace("\n", " "),
        # Latency
        "latency_ms":       latency_ms,
        # Strict Accuracy
        "predicted_action": acc["predicted_action"],
        "exact_match":      acc["exact_match"],
        "in_accepted_set":  acc["in_accepted_set"],
        # Faithfulness
        "faithfulness_rule": faith_rule["score"],
        "hallucinated_nums": str(faith_rule["hallucinated"]),
        "faithfulness_llm":  faith_llm.get("score"),
        "faithfulness_verdict": faith_llm.get("verdict"),
        # Relevancy
        "answer_relevancy":  relevancy,
        # Qualitative
        "qual_accuracy":     qual.get("qual_accuracy"),
        "qual_clarity":      qual.get("qual_clarity"),
        "qual_usefulness":   qual.get("qual_usefulness"),
        "justification":     qual.get("justification", ""),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_evaluation(
    model_type:    str  = "dqn",
    n_normal:      int  = 5,
    n_attack:      int  = 8,
    n_noisy:       int  = 3,
    use_llm_judge: bool = True,
    output:        str  = "logs/llm_auto_eval.csv",
) -> list[dict]:

    # ── In training summary trước ────────────────────────────────────────────
    print("=" * 60)
    print(f"  LLM EVALUATOR — model={model_type.upper()}")
    print("=" * 60)
    try:
        summary = get_training_summary()
        print("\n[Training Summary]")
        print(summary[["model","seed","best_reward","mean_reward_last100","total_episodes"]]
              .to_string(index=False))
    except Exception as e:
        print(f"  [WARN] Không đọc được training summary: {e}")

    # ── Lấy dữ liệu thật từ model replay ────────────────────────────────────
    print(f"\n[Data Loading] Lấy dữ liệu từ {model_type.upper()} replay...")
    decisions = sample_decisions(model_type, n_normal=n_normal,
                                 n_attack=n_attack, n_noisy=n_noisy)

    # Export để audit
    replay_csv = os.path.join(ROOT, "logs", f"{model_type}_replay_eval.csv")
    all_recs = decisions["normal"] + decisions["attack"] + decisions["noisy"]
    export_replay_to_csv(all_recs, replay_csv)

    # ── Chạy đánh giá ────────────────────────────────────────────────────────
    records = []
    for scenario, recs in decisions.items():
        if not recs:
            print(f"  [SKIP] Không có dữ liệu cho scenario '{scenario}'")
            continue

        print(f"\n{'─'*60}")
        print(f"  Scenario: {scenario.upper()} ({len(recs)} records)")
        print(f"{'─'*60}")

        for i, rec in enumerate(recs):
            print(f"  [{i+1}/{len(recs)}] step={rec['step']} "
                  f"action={rec['action_name']} "
                  f"reward={rec['reward']:.3f} "
                  f"attack={rec['attack_indicator']:.2f}")
            try:
                result = evaluate_record(rec, scenario, use_llm_judge)
                result["scenario_idx"] = i
                records.append(result)
                print(f"           → LLM latency={result['latency_ms']:.0f}ms "
                      f"faith={result['faithfulness_rule']:.3f} "
                      f"exact={result['exact_match']}")
            except Exception as e:
                print(f"           → [ERROR] {e}")

    if not records:
        print("Không có kết quả nào. Kết thúc.")
        return []

    # ── Lưu CSV ─────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"\n  Kết quả đã lưu → {output}")

    _print_summary(records)
    return records


def _print_summary(records: list[dict]):
    print("\n" + "═" * 72)
    print("  KẾT QUẢ ĐÁNH GIÁ TỔNG HỢP")
    print("═" * 72)

    for scenario in ["normal", "attack", "noisy"]:
        recs = [r for r in records if r["scenario"] == scenario]
        if not recs:
            continue

        exact_rate = sum(1 for r in recs if r["exact_match"]) / len(recs)
        avg_faith  = sum(r["faithfulness_rule"] for r in recs
                         if r["faithfulness_rule"] is not None) / max(1, len(recs))
        avg_lat    = sum(r["latency_ms"] for r in recs) / len(recs)
        avg_rel    = sum(r["answer_relevancy"] for r in recs) / len(recs)

        print(f"\n  [{scenario.upper()}] — {len(recs)} mẫu")
        print(f"    Strict Accuracy    : {exact_rate:.1%}")
        print(f"    Faithfulness (rule): {avg_faith:.3f}")
        print(f"    Latency trung bình : {avg_lat:.0f} ms")
        print(f"    Answer Relevancy   : {avg_rel:.3f}")

        qa = [r["qual_accuracy"]   for r in recs if r.get("qual_accuracy")]
        qc = [r["qual_clarity"]    for r in recs if r.get("qual_clarity")]
        qu = [r["qual_usefulness"] for r in recs if r.get("qual_usefulness")]
        if qa: print(f"    Qual Accuracy      : {sum(qa)/len(qa):.2f}/5")
        if qc: print(f"    Qual Clarity       : {sum(qc)/len(qc):.2f}/5")
        if qu: print(f"    Qual Usefulness    : {sum(qu)/len(qu):.2f}/5")

    print("═" * 72)


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Evaluator — dữ liệu từ RL training thật")
    parser.add_argument("--model",        default="dqn",  choices=["dqn","ppo","best"])
    parser.add_argument("--n-normal",     type=int, default=5)
    parser.add_argument("--n-attack",     type=int, default=8)
    parser.add_argument("--n-noisy",      type=int, default=3)
    parser.add_argument("--no-llm-judge", action="store_true",
                        help="Bỏ qua LLM-as-Judge (tiết kiệm API calls)")
    parser.add_argument("--output",       default="logs/llm_auto_eval.csv")
    args = parser.parse_args()

    run_evaluation(
        model_type    = args.model,
        n_normal      = args.n_normal,
        n_attack      = args.n_attack,
        n_noisy       = args.n_noisy,
        use_llm_judge = not args.no_llm_judge,
        output        = args.output,
    )