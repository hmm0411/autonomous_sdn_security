#!/usr/bin/env bash

set -uo pipefail

REPO="$HOME/autonomous_sdn_security"
NS="sdn-security"

# =============================
# Benchmark timing
# Có thể chỉnh nhanh khi chạy:
# WARMUP_SECONDS=20 ATTACK_SECONDS=60 RECOVERY_SECONDS=15 ./scripts/run_benchmark_36.sh
# =============================
WARMUP_SECONDS="${WARMUP_SECONDS:-30}"
ATTACK_SECONDS="${ATTACK_SECONDS:-90}"
RECOVERY_SECONDS="${RECOVERY_SECONDS:-30}"
MN_STARTUP_SECONDS="${MN_STARTUP_SECONDS:-12}"

# ACTION_DRY_RUN=true: benchmark quyết định/action logic an toàn
# ACTION_DRY_RUN=false: áp dụng action thật lên controller
ACTION_DRY_RUN="${ACTION_DRY_RUN:-true}"

INTENSITY="${INTENSITY:-medium}"
SURROGATE_MODEL="${SURROGATE_MODEL:-models/surrogate_model.pkl}"

BENCH_DIR="$REPO/results/benchmark_36"
MAIN_LOG="$BENCH_DIR/benchmark_36_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$BENCH_DIR" "$REPO/logs" "$REPO/results/evaluation"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$MAIN_LOG"
}

attack_cmd() {
  case "$1" in
    normal) echo "" ;;
    ddos_flood) echo "py net.manager.ddos_flood()" ;;
    flow_overflow) echo "py net.manager.flow_overflow()" ;;
    packet_in_flood) echo "py net.manager.packet_in_flood()" ;;
    ip_spoofing) echo "py net.manager.ip_spoofing()" ;;
    port_scanning) echo "py net.manager.port_scanning()" ;;
    *)
      echo ""
      ;;
  esac
}

get_control_loop_pod() {
  kubectl get pods -n "$NS" --sort-by=.metadata.creationTimestamp \
    | awk '/control-loop/ && $3=="Running" {p=$1} END{print p}'
}

cleanup_mininet() {
  log "[MININET] sudo mn -c"
  sudo -v
  sudo mn -c >> "$MAIN_LOG" 2>&1 || true
}

send_mn_cmd() {
  local fd="$1"
  local cmd="$2"

  log "[MININET_CMD] $cmd"
  printf "%s\n" "$cmd" >&"$fd" 2>/dev/null || true
}

start_mininet() {
  local tag="$1"
  local mn_log="$BENCH_DIR/${tag}_mininet.log"

  cleanup_mininet

  log "[MININET] starting traffic_generator/run.py | log=$mn_log"

  sudo -v
  coproc MNPROC {
    cd "$REPO/traffic_generator" && sudo -n python3 run.py >> "$mn_log" 2>&1
  }

  MN_PID="$MNPROC_PID"
  MN_IN="${MNPROC[1]}"

  sleep "$MN_STARTUP_SECONDS"

  if ! kill -0 "$MN_PID" 2>/dev/null; then
    log "[MININET_ERROR] run.py exited too early. Check $mn_log"
    return 1
  fi

  send_mn_cmd "$MN_IN" "pingall"
  sleep 5

  return 0
}

stop_mininet_process() {
  local fd="${MN_IN:-}"
  local pid="${MN_PID:-}"

  if [[ -n "$fd" ]]; then
    exec {fd}>&- 2>/dev/null || true
  fi

  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    sleep 3
    kill "$pid" 2>/dev/null || true
    sleep 2
    kill -9 "$pid" 2>/dev/null || true
  fi

  wait "$pid" 2>/dev/null || true

  cleanup_mininet
}

run_mininet_attack_phase() {
  local attack="$1"
  local run_id="$2"

  local cmd
  cmd="$(attack_cmd "$attack")"

  start_mininet "${run_id}_attack" || {
    log "[SESSION_ERROR] Cannot start Mininet for $run_id"
    return 1
  }

  log "[PHASE] warmup ${WARMUP_SECONDS}s | run_id=$run_id"
  sleep "$WARMUP_SECONDS"

  if [[ "$attack" == "normal" ]]; then
    log "[PHASE] normal traffic ${ATTACK_SECONDS}s | run_id=$run_id"
    sleep "$ATTACK_SECONDS"
    send_mn_cmd "$MN_IN" "exit"
  else
    log "[PHASE] attack=$attack ${ATTACK_SECONDS}s | run_id=$run_id"
    send_mn_cmd "$MN_IN" "$cmd"
    sleep "$ATTACK_SECONDS"

    # Lưu ý: lệnh này có thể làm Mininet tự out.
    send_mn_cmd "$MN_IN" "py net.manager.stop_all()"
    sleep 5
  fi

  stop_mininet_process
  return 0
}

run_mininet_recovery_phase() {
  local run_id="$1"

  if [[ "$RECOVERY_SECONDS" == "0" ]]; then
    return 0
  fi

  start_mininet "${run_id}_recovery" || {
    log "[RECOVERY_WARN] Cannot start recovery Mininet for $run_id"
    return 0
  }

  log "[PHASE] recovery ${RECOVERY_SECONDS}s | run_id=$run_id"
  sleep "$RECOVERY_SECONDS"

  send_mn_cmd "$MN_IN" "exit"
  stop_mininet_process
}

configure_control_loop() {
  local attack="$1"
  local cfg_name="$2"
  local mode="$3"
  local model="$4"
  local run_id="$5"

  log "[K8S] configure control-loop | run_id=$run_id mode=$mode model=$model attack=$attack"

  kubectl set env deployment/control-loop -n "$NS" \
    MODE="$mode" \
    MODEL_TYPE="$model" \
    ACTION_DRY_RUN="$ACTION_DRY_RUN" \
    STRICT_RL=true \
    ATTACK_TYPE="$attack" \
    ATTACK_INTENSITY="$INTENSITY" \
    RUN_ID="$run_id" \
    SURROGATE_MODEL="$SURROGATE_MODEL" >> "$MAIN_LOG" 2>&1

  kubectl rollout restart deployment/control-loop -n "$NS" >> "$MAIN_LOG" 2>&1
  kubectl rollout status deployment/control-loop -n "$NS" --timeout=180s | tee -a "$MAIN_LOG"

  local pod
  pod="$(get_control_loop_pod)"

  if [[ -z "$pod" ]]; then
    log "[K8S_ERROR] Cannot find Running control-loop pod"
    return 1
  fi

  log "[K8S] control-loop pod=$pod"

  kubectl exec "$pod" -n "$NS" -- printenv \
    | egrep 'MODE|MODEL_TYPE|RUN_ID|ATTACK_TYPE|ATTACK_INTENSITY|ACTION_DRY_RUN' \
    | tee -a "$MAIN_LOG" || true

  # Lưu log control-loop theo từng session
  kubectl logs -f "$pod" -n "$NS" > "$BENCH_DIR/${run_id}_control_loop.log" 2>&1 &
  CONTROL_LOG_PID=$!

  sleep 5
  return 0
}

stop_control_loop_log() {
  if [[ -n "${CONTROL_LOG_PID:-}" ]]; then
    kill "$CONTROL_LOG_PID" 2>/dev/null || true
    wait "$CONTROL_LOG_PID" 2>/dev/null || true
  fi
}

post_summary() {
  log "[POST] summarize runtime_eval.csv"

  cd "$REPO"

  python3 - <<'PY'
import pandas as pd
from pathlib import Path

runtime = Path("logs/runtime_eval.csv")
outdir = Path("results/evaluation")
outdir.mkdir(parents=True, exist_ok=True)

if not runtime.exists():
    print("[POST_ERROR] logs/runtime_eval.csv not found")
    raise SystemExit(1)

df = pd.read_csv(runtime, low_memory=False)

# ép kiểu các cột số cần thiết
for c in ["packet_rate", "byte_rate", "flow_count", "latency", "packet_loss",
          "controller_cpu", "action", "reward", "reward_staging"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

print("Rows:", len(df))
print("\nRun IDs:")
print(df["run_id"].value_counts())

print("\nAction distribution:")
print(df.groupby(["run_id", "attack_type", "mode", "model"])["action"]
        .value_counts()
        .sort_index())

summary = (
    df.groupby(["run_id", "attack_type", "intensity", "mode", "model"], dropna=False)
      .agg(
          rows=("action", "size"),
          action_0=("action", lambda s: int((s == 0).sum())),
          action_1=("action", lambda s: int((s == 1).sum())),
          action_2=("action", lambda s: int((s == 2).sum())),
          action_3=("action", lambda s: int((s == 3).sum())),
          action_4=("action", lambda s: int((s == 4).sum())),
          avg_packet_rate=("packet_rate", "mean"),
          avg_latency=("latency", "mean"),
          avg_packet_loss=("packet_loss", "mean"),
          avg_controller_cpu=("controller_cpu", "mean"),
          avg_reward=("reward", "mean"),
      )
      .reset_index()
)

summary_path = outdir / "benchmark_36_summary.csv"
summary.to_csv(summary_path, index=False)
print("\nSaved:", summary_path)
print(summary)
PY

  log "[POST] run experiments.summarize_metrics"
  python3 -m experiments.summarize_metrics >> "$MAIN_LOG" 2>&1 || \
    log "[POST_WARN] experiments.summarize_metrics failed, but benchmark_36_summary.csv was created"
}

main() {
  cd "$REPO"

  log "===== START BENCHMARK 36 ====="
  log "REPO=$REPO"
  log "ACTION_DRY_RUN=$ACTION_DRY_RUN"
  log "WARMUP_SECONDS=$WARMUP_SECONDS ATTACK_SECONDS=$ATTACK_SECONDS RECOVERY_SECONDS=$RECOVERY_SECONDS"
  log "INTENSITY=$INTENSITY"

  sudo -v

  # Backup và xóa runtime log cũ để tránh lỗi schema cũ/mới
  if [[ -f logs/runtime_eval.csv ]]; then
    cp logs/runtime_eval.csv "logs/runtime_eval_backup_before_bench36_$(date +%Y%m%d_%H%M%S).csv"
  fi

  rm -f logs/runtime_eval.csv

  # Không xóa transition_log.csv mặc định để giữ data Twin.
  # Nếu muốn benchmark sạch transition thì tự bật biến CLEAR_TRANSITION=true.
  if [[ "${CLEAR_TRANSITION:-false}" == "true" ]]; then
    cp logs/transition_log.csv "logs/transition_log_backup_before_bench36_$(date +%Y%m%d_%H%M%S).csv" 2>/dev/null || true
    rm -f logs/transition_log.csv
  fi

  # Xóa pod lỗi cũ
  kubectl get pods -n "$NS" | awk '/Error|Evicted|ContainerStatusUnknown/ {print $1}' | \
    xargs -r kubectl delete pod -n "$NS" --force --grace-period=0 >> "$MAIN_LOG" 2>&1 || true

  cleanup_mininet

  ATTACKS=(
    normal
    ddos_flood
    flow_overflow
    packet_in_flood
    ip_spoofing
    port_scanning
  )

  CONFIGS=(
    "no_defense|no_defense|dqn"
    "rule|rule|dqn"
    "rl_dqn|rl|dqn"
    "rl_ppo|rl|ppo"
    "rl_twin_dqn|rl_twin|dqn"
    "rl_twin_ppo|rl_twin|ppo"
  )

  total=$(( ${#ATTACKS[@]} * ${#CONFIGS[@]} ))
  idx=0

  for attack in "${ATTACKS[@]}"; do
    for cfg in "${CONFIGS[@]}"; do
      IFS='|' read -r cfg_name mode model <<< "$cfg"

      idx=$((idx + 1))
      run_id="bench36_${attack}_${cfg_name}_r1"

      log "===== SESSION $idx/$total | $run_id ====="

      stop_control_loop_log || true

      if ! configure_control_loop "$attack" "$cfg_name" "$mode" "$model" "$run_id"; then
        log "[SESSION_SKIP] control-loop configure failed | $run_id"
        continue
      fi

      if ! run_mininet_attack_phase "$attack" "$run_id"; then
        log "[SESSION_WARN] attack phase failed | $run_id"
      fi

      run_mininet_recovery_phase "$run_id"

      stop_control_loop_log || true

      log "===== DONE SESSION $idx/$total | $run_id ====="
      sleep 5
    done
  done

  cleanup_mininet
  post_summary

  log "===== FINISHED BENCHMARK 36 ====="
  log "Main log: $MAIN_LOG"
  log "Runtime CSV: $REPO/logs/runtime_eval.csv"
  log "Summary CSV: $REPO/results/evaluation/benchmark_36_summary.csv"
}

main "$@"
