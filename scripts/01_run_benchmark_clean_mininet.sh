#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# CONFIG
# =========================================================
REPO="${REPO:-$HOME/autonomous_sdn_security}"
NS="${NS:-sdn-security}"
CONTROL_DEPLOY="${CONTROL_DEPLOY:-control-loop}"
MN_SESSION="${MN_SESSION:-mn}"

WARMUP_SECONDS="${WARMUP_SECONDS:-30}"
ATTACK_SECONDS="${ATTACK_SECONDS:-90}"
RECOVERY_SECONDS="${RECOVERY_SECONDS:-30}"
MININET_BOOT_SECONDS="${MININET_BOOT_SECONDS:-18}"
RUNS="${RUNS:-1}"

ACTION_DRY_RUN="${ACTION_DRY_RUN:-true}"
STRICT_RL="${STRICT_RL:-true}"

cd "$REPO"
export PYTHONPATH="$PWD"

sudo -v

# =========================================================
# ATTACKS
# =========================================================
ATTACKS=(
  "normal"
  "ddos_flood"
  "flow_overflow"
  "packet_in_flood"
  "ip_spoofing"
  "port_scanning"
)

# =========================================================
# CONFIGS
# format:
# eval_config|mode|model|guard|twin|llm
# =========================================================
CONFIGS=(
  "no_defense|no_defense|dqn|false|false|false"
  "rule|rule|dqn|false|false|false"
  "rl_dqn|rl|dqn|false|false|false"
  "rl_ppo|rl|ppo|false|false|false"
  "rl_guard_ppo|rl|ppo|true|false|false"
  "rl_twin_ppo|rl_twin|ppo|true|true|false"
  "full_system_ppo|full|ppo|true|true|true"
)

# Nếu chưa muốn chạy LLM để tiết kiệm thời gian/API,
# comment dòng full_system_ppo ở trên.

# =========================================================
# HELPER FUNCTIONS
# =========================================================
function attack_cmd() {
    local attack="$1"

    case "$attack" in
        normal)
            echo ""
            ;;
        ddos_flood)
            echo "py net.manager.ddos_flood()"
            ;;
        flow_overflow)
            echo "py net.manager.flow_overflow()"
            ;;
        packet_in_flood)
            echo "py net.manager.packet_in_flood()"
            ;;
        ip_spoofing)
            echo "py net.manager.ip_spoofing()"
            ;;
        port_scanning)
            echo "py net.manager.port_scanning()"
            ;;
        *)
            echo ""
            ;;
    esac
}

function cleanup_mininet() {
    echo
    echo "========== CLEANUP MININET =========="

    if tmux has-session -t "$MN_SESSION" 2>/dev/null; then
        echo "[INFO] sending stop_all to Mininet"
        tmux send-keys -t "$MN_SESSION" "py net.manager.stop_all()" C-m || true
        sleep 2

        echo "[INFO] exiting Mininet CLI"
        tmux send-keys -t "$MN_SESSION" "exit" C-m || true
        sleep 2

        echo "[INFO] killing tmux session $MN_SESSION"
        tmux kill-session -t "$MN_SESSION" || true
        sleep 1
    fi

    echo "[INFO] running sudo mn -c"
    sudo mn -c || true
    sleep 2
}

function start_mininet() {
    echo
    echo "========== START MININET =========="

    cleanup_mininet

    tmux new-session -d -s "$MN_SESSION" "cd '$REPO/traffic_generator' && sudo -E python3 run.py"

    echo "[INFO] waiting Mininet boot ${MININET_BOOT_SECONDS}s"
    sleep "$MININET_BOOT_SECONDS"

    echo "[INFO] pingall sanity check"
    tmux send-keys -t "$MN_SESSION" "pingall" C-m || true
    sleep 5
}

function mn_send() {
    local cmd="$1"

    if [ -z "$cmd" ]; then
        return 0
    fi

    if ! tmux has-session -t "$MN_SESSION" 2>/dev/null; then
        echo "[ERROR] Mininet tmux session does not exist"
        return 1
    fi

    echo "[MN] $cmd"
    tmux send-keys -t "$MN_SESSION" "$cmd" C-m
}

function set_control_env() {
    local eval_config="$1"
    local mode="$2"
    local model="$3"
    local guard="$4"
    local twin="$5"
    local llm="$6"
    local attack="$7"
    local phase="$8"
    local run_id="$9"

    kubectl -n "$NS" set env deployment/"$CONTROL_DEPLOY" \
        EVAL_CONFIG="$eval_config" \
        MODE="$mode" \
        MODEL_TYPE="$model" \
        ENABLE_GUARD="$guard" \
        ENABLE_TWIN="$twin" \
        ENABLE_LLM="$llm" \
        ATTACK_TYPE="$attack" \
        ATTACK_INTENSITY=medium \
        RUN_ID="$run_id" \
        PHASE="$phase" \
        ACTION_DRY_RUN="$ACTION_DRY_RUN" \
        STRICT_RL="$STRICT_RL" \
        ALLOW_UNTRAINED_FALLBACK=false \
        RUNTIME_LOG=/app/logs/runtime_eval.csv \
        TRANSITION_LOG=/app/logs/transition_log.csv \
        SURROGATE_MODEL=models/surrogate_model.pkl
}

function restart_control_loop() {
    kubectl -n "$NS" rollout restart deployment/"$CONTROL_DEPLOY"
    kubectl -n "$NS" rollout status deployment/"$CONTROL_DEPLOY" --timeout=120s
    sleep 3
}

function run_one_case() {
    local attack="$1"
    local run_id="$2"
    local config_line="$3"

    IFS="|" read -r eval_config mode model guard twin llm <<< "$config_line"

    echo
    echo "========================================================="
    echo "[CASE] run=$run_id attack=$attack config=$eval_config"
    echo "       mode=$mode model=$model guard=$guard twin=$twin llm=$llm"
    echo "========================================================="

    # Mỗi case tạo Mininet mới cho sạch.
    start_mininet

    # -------------------------
    # WARMUP
    # -------------------------
    echo
    echo "[PHASE] WARMUP ${WARMUP_SECONDS}s"

    set_control_env "$eval_config" "$mode" "$model" "$guard" "$twin" "$llm" "$attack" "warmup" "$run_id"
    restart_control_loop

    mn_send "py net.manager.stop_all()" || true
    sleep "$WARMUP_SECONDS"

    # -------------------------
    # ATTACK
    # -------------------------
    echo
    echo "[PHASE] ATTACK ${ATTACK_SECONDS}s"

    set_control_env "$eval_config" "$mode" "$model" "$guard" "$twin" "$llm" "$attack" "attack" "$run_id"
    restart_control_loop

    local cmd
    cmd="$(attack_cmd "$attack")"

    if [ -n "$cmd" ]; then
        mn_send "$cmd"
    else
        echo "[INFO] normal traffic: no attack command"
    fi

    sleep "$ATTACK_SECONDS"

    # -------------------------
    # RECOVERY
    # -------------------------
    echo
    echo "[PHASE] RECOVERY ${RECOVERY_SECONDS}s"

    set_control_env "$eval_config" "$mode" "$model" "$guard" "$twin" "$llm" "$attack" "recovery" "$run_id"
    restart_control_loop

    mn_send "py net.manager.stop_all()" || true
    sleep "$RECOVERY_SECONDS"

    # Cleanup sau mỗi case như bạn yêu cầu.
    cleanup_mininet
}

function copy_logs_from_pod_if_needed() {
    echo
    echo "========== COPY LOGS FROM POD IF NEEDED =========="

    mkdir -p "$REPO/logs"

    local pod
    pod="$(kubectl -n "$NS" get pods -l app=control-loop -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)"

    if [ -z "$pod" ]; then
        echo "[WARN] cannot find control-loop pod by label app=control-loop"
        return 0
    fi

    kubectl -n "$NS" cp "$pod:/app/logs/runtime_eval.csv" "$REPO/logs/runtime_eval.csv" 2>/dev/null || true
    kubectl -n "$NS" cp "$pod:/app/logs/transition_log.csv" "$REPO/logs/transition_log.csv" 2>/dev/null || true
}

# =========================================================
# MAIN
# =========================================================
trap cleanup_mininet EXIT

echo
echo "========== BENCHMARK START =========="
echo "[INFO] RUNS=$RUNS"
echo "[INFO] WARMUP_SECONDS=$WARMUP_SECONDS"
echo "[INFO] ATTACK_SECONDS=$ATTACK_SECONDS"
echo "[INFO] RECOVERY_SECONDS=$RECOVERY_SECONDS"
echo "[INFO] ACTION_DRY_RUN=$ACTION_DRY_RUN"

for run_id in $(seq 1 "$RUNS"); do
    for attack in "${ATTACKS[@]}"; do
        for config in "${CONFIGS[@]}"; do
            run_one_case "$attack" "$run_id" "$config"
        done
    done
done

copy_logs_from_pod_if_needed

echo
echo "========== SUMMARY =========="
if [ -f logs/runtime_eval.csv ]; then
    echo "[INFO] runtime rows:"
    wc -l logs/runtime_eval.csv || true
else
    echo "[ERROR] logs/runtime_eval.csv not found"
fi

echo
echo "[DONE] Benchmark completed."
