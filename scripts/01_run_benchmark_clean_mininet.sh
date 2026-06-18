#!/usr/bin/env bash
set -uo pipefail

REPO="${REPO:-$HOME/autonomous_sdn_security}"
NS="${NS:-sdn-security}"
CONTROL_DEPLOY="${CONTROL_DEPLOY:-control-loop}"
MN_SESSION="${MN_SESSION:-mn}"

WARMUP_SECONDS="${WARMUP_SECONDS:-20}"
ATTACK_SECONDS="${ATTACK_SECONDS:-40}"
RECOVERY_SECONDS="${RECOVERY_SECONDS:-20}"
MININET_BOOT_SECONDS="${MININET_BOOT_SECONDS:-20}"
RUNS="${RUNS:-1}"

ACTION_DRY_RUN="${ACTION_DRY_RUN:-true}"
STRICT_RL="${STRICT_RL:-true}"

cd "$REPO" || exit 1
export PYTHONPATH="$PWD"

mkdir -p logs results/evaluation

PROGRESS_LOG="logs/benchmark_progress_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$PROGRESS_LOG") 2>&1

sudo -v

ATTACKS=(
  "normal"
  "ddos_flood"
  "flow_overflow"
  "packet_in_flood"
  "ip_spoofing"
  "port_scanning"
)

CONFIGS=(
  "no_defense|no_defense|dqn|false|false|false"
  "rule|rule|dqn|false|false|false"
  "rl_dqn|rl|dqn|false|false|false"
  "rl_ppo|rl|ppo|false|false|false"
  "rl_guard_ppo|rl|ppo|true|false|false"
  "rl_twin_ppo|rl_twin|ppo|true|true|false"
  "full_system_ppo|full|ppo|true|true|true"
)

function now() {
    date "+%Y-%m-%d %H:%M:%S"
}

function log() {
    echo "[$(now)] $*"
}

function attack_cmd() {
    local attack="$1"

    case "$attack" in
        normal)
            echo "py net.manager.normal_medium()"
            ;;
        ddos_flood)
            echo "py net.manager.ddos_flood(num_attackers=3, intensity='medium')"
            ;;
        flow_overflow)
            echo "py net.manager.flow_overflow(num_attackers=3, flows_per_attacker=500)"
            ;;
        packet_in_flood)
            echo "py net.manager.packet_in_flood(num_attackers=3, intensity='medium')"
            ;;
        ip_spoofing)
            echo "py net.manager.ip_spoofing(num_attackers=2, intensity='medium')"
            ;;
        port_scanning)
            echo "py net.manager.port_scanning(attacker_index=0, start_port=1, end_port=1000)"
            ;;
        *)
            echo ""
            ;;
    esac
}

function cleanup_mininet() {
    log "CLEANUP MININET"

    if tmux has-session -t "$MN_SESSION" 2>/dev/null; then
        tmux send-keys -t "$MN_SESSION" "py net.manager.stop_all()" C-m || true
        sleep 2
        tmux send-keys -t "$MN_SESSION" "exit" C-m || true
        sleep 2
        tmux kill-session -t "$MN_SESSION" || true
        sleep 1
    fi

    sudo mn -c || true
    sleep 2
}

function reset_control_loop_idle() {
    log "RESET CONTROL LOOP TO IDLE/COLLECT"

    kubectl -n "$NS" set env deployment/"$CONTROL_DEPLOY" \
        MODE=collect \
        MODEL_TYPE=dqn \
        EVAL_CONFIG=collect \
        PHASE=idle \
        ATTACK_TYPE=normal \
        ATTACK_INTENSITY=none \
        RUN_ID=0 \
        ENABLE_GUARD=false \
        ENABLE_TWIN=false \
        ENABLE_LLM=false \
        ACTION_DRY_RUN=true \
        STRICT_RL=true \
        RUNTIME_LOG=/app/logs/runtime_eval.csv \
        TRANSITION_LOG=/app/logs/transition_log.csv \
        >/dev/null 2>&1 || true

    kubectl -n "$NS" rollout restart deployment/"$CONTROL_DEPLOY" >/dev/null 2>&1 || true
}

function final_cleanup() {
    log "FINAL CLEANUP"
    cleanup_mininet
    reset_control_loop_idle
}

trap final_cleanup EXIT

function start_mininet() {
    log "START MININET"

    cleanup_mininet

    tmux new-session -d -s "$MN_SESSION" \
        "cd '$REPO/traffic_generator' && sudo -E python3 run.py"

    log "Waiting Mininet boot ${MININET_BOOT_SECONDS}s"
    sleep "$MININET_BOOT_SECONDS"

    tmux send-keys -t "$MN_SESSION" "pingall" C-m || true
    sleep 5

    log "Mininet pane tail:"
    tmux capture-pane -t "$MN_SESSION" -p | tail -20 || true
}

function mn_send() {
    local cmd="$1"

    if [ -z "$cmd" ]; then
        return 0
    fi

    if ! tmux has-session -t "$MN_SESSION" 2>/dev/null; then
        log "ERROR: Mininet tmux session does not exist"
        return 1
    fi

    log "MN CMD: $cmd"
    tmux send-keys -t "$MN_SESSION" "$cmd" C-m
    sleep 2

    log "Mininet pane tail after command:"
    tmux capture-pane -t "$MN_SESSION" -p | tail -20 || true
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

    log "SET ENV config=$eval_config attack=$attack phase=$phase run=$run_id"

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

    return $?
}

function restart_control_loop() {
    log "RESTART CONTROL LOOP"

    kubectl -n "$NS" scale deployment/"$CONTROL_DEPLOY" --replicas=1 || return 1
    kubectl -n "$NS" rollout restart deployment/"$CONTROL_DEPLOY" || return 1
    kubectl -n "$NS" rollout status deployment/"$CONTROL_DEPLOY" --timeout=120s || return 1

    sleep 3

    log "control-loop logs tail:"
    kubectl -n "$NS" logs deployment/"$CONTROL_DEPLOY" --tail=20 || true
}

function count_rows() {
    if [ -f logs/runtime_eval.csv ]; then
        wc -l logs/runtime_eval.csv | awk '{print $1}'
    else
        echo 0
    fi
}

function run_one_case() {
    local attack="$1"
    local run_id="$2"
    local config_line="$3"

    IFS="|" read -r eval_config mode model guard twin llm <<< "$config_line"

    log "========================================================="
    log "CASE START run=$run_id attack=$attack config=$eval_config"
    log "mode=$mode model=$model guard=$guard twin=$twin llm=$llm"
    log "========================================================="

    local rows_before
    rows_before="$(count_rows)"

    start_mininet || return 1

    log "PHASE WARMUP ${WARMUP_SECONDS}s"
    set_control_env "$eval_config" "$mode" "$model" "$guard" "$twin" "$llm" "$attack" "warmup" "$run_id" || return 1
    restart_control_loop || return 1
    mn_send "py net.manager.stop_all()" || true
    sleep "$WARMUP_SECONDS"

    log "PHASE ATTACK ${ATTACK_SECONDS}s"
    set_control_env "$eval_config" "$mode" "$model" "$guard" "$twin" "$llm" "$attack" "attack" "$run_id" || return 1
    restart_control_loop || return 1

    local cmd
    cmd="$(attack_cmd "$attack")"

    if [ -n "$cmd" ]; then
        mn_send "$cmd" || return 1
    else
        log "normal traffic: no attack command"
    fi

    sleep "$ATTACK_SECONDS"

    log "PHASE RECOVERY ${RECOVERY_SECONDS}s"
    set_control_env "$eval_config" "$mode" "$model" "$guard" "$twin" "$llm" "$attack" "recovery" "$run_id" || return 1
    restart_control_loop || return 1
    mn_send "py net.manager.stop_all()" || true
    sleep "$RECOVERY_SECONDS"

    cleanup_mininet

    local rows_after
    rows_after="$(count_rows)"

    log "CASE END run=$run_id attack=$attack config=$eval_config rows_before=$rows_before rows_after=$rows_after"
    return 0
}

log "========== BENCHMARK START =========="
log "RUNS=$RUNS"
log "WARMUP_SECONDS=$WARMUP_SECONDS"
log "ATTACK_SECONDS=$ATTACK_SECONDS"
log "RECOVERY_SECONDS=$RECOVERY_SECONDS"
log "ACTION_DRY_RUN=$ACTION_DRY_RUN"
log "PROGRESS_LOG=$PROGRESS_LOG"

kubectl -n "$NS" scale deployment/"$CONTROL_DEPLOY" --replicas=1 || true

FAILED_CASES=0
PASSED_CASES=0

for run_id in $(seq 1 "$RUNS"); do
    for attack in "${ATTACKS[@]}"; do
        for config in "${CONFIGS[@]}"; do
            if run_one_case "$attack" "$run_id" "$config"; then
                PASSED_CASES=$((PASSED_CASES + 1))
                log "CASE PASSED attack=$attack config=$config"
            else
                FAILED_CASES=$((FAILED_CASES + 1))
                log "CASE FAILED attack=$attack config=$config"
                cleanup_mininet
                reset_control_loop_idle
                sleep 5
            fi
        done
    done
done

log "========== BENCHMARK DONE =========="
log "PASSED_CASES=$PASSED_CASES"
log "FAILED_CASES=$FAILED_CASES"

if [ -f logs/runtime_eval.csv ]; then
    log "runtime rows:"
    wc -l logs/runtime_eval.csv || true
else
    log "ERROR: logs/runtime_eval.csv not found"
fi

log "Progress log saved at: $PROGRESS_LOG"
