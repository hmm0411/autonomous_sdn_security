#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# CONFIG
# =========================================================
REPO="${REPO:-$HOME/autonomous_sdn_security}"
NS="${NS:-sdn-security}"
IMAGE="${IMAGE:-hmm0411/sdn-rl-agent:state8}"

CONTROL_DEPLOY="${CONTROL_DEPLOY:-control-loop}"
CONTROL_CONTAINER="${CONTROL_CONTAINER:-loop}"

DQN_DEPLOY="${DQN_DEPLOY:-rl-serving-dqn}"
DQN_CONTAINER="${DQN_CONTAINER:-serving}"

PPO_DEPLOY="${PPO_DEPLOY:-rl-serving-ppo}"
PPO_CONTAINER="${PPO_CONTAINER:-serving}"

NO_CACHE="${NO_CACHE:-0}"
DEEP_CLEAN="${DEEP_CLEAN:-0}"

cd "$REPO"
export PYTHONPATH="$PWD"

echo "========================================================="
echo "[INFO] REPO=$REPO"
echo "[INFO] NS=$NS"
echo "[INFO] IMAGE=$IMAGE"
echo "========================================================="

sudo -v

# =========================================================
# 0. CHECK DISK BEFORE CLEAN
# =========================================================
echo
echo "========== DISK BEFORE CLEAN =========="
df -h || true
echo
du -sh "$REPO" 2>/dev/null || true
du -sh "$REPO/logs" "$REPO/results" "$REPO/runs" "$REPO/mlruns" "$REPO/models" 2>/dev/null || true
sudo du -sh /var/lib/docker /var/lib/containerd /var/lib/rancher/k3s 2>/dev/null || true

# =========================================================
# 1. STOP MININET CLEANLY
# =========================================================
echo
echo "========== CLEAN MININET =========="

if tmux has-session -t mn 2>/dev/null; then
    echo "[INFO] stopping tmux session mn"
    tmux send-keys -t mn "py net.manager.stop_all()" C-m || true
    sleep 2
    tmux send-keys -t mn "exit" C-m || true
    sleep 2
    tmux kill-session -t mn || true
fi

sudo mn -c || true

# =========================================================
# 2. BACKUP OLD RESULTS BEFORE DELETE
# =========================================================
echo
echo "========== BACKUP OLD RESULTS =========="

TS="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="$REPO/_backup/$TS"
mkdir -p "$BACKUP_DIR"

if [ -d logs ]; then cp -a logs "$BACKUP_DIR/logs" || true; fi
if [ -d results ]; then cp -a results "$BACKUP_DIR/results" || true; fi

echo "[INFO] backup saved to $BACKUP_DIR"

# =========================================================
# 3. DELETE OLD K8S RESOURCES
# =========================================================
echo
echo "========== DELETE OLD K8S RESOURCES =========="

kubectl delete ns "$NS" --ignore-not-found --wait=true || true
sleep 3
kubectl create ns "$NS"

# =========================================================
# 4. CLEAN LOCAL LOGS / RESULTS
# =========================================================
echo
echo "========== CLEAN LOCAL LOGS / RESULTS =========="

rm -f logs/runtime_eval.csv logs/transition_log.csv
rm -rf results/evaluation results/digital_twin
mkdir -p logs results/evaluation results/digital_twin models runs

# Không xóa models/*.pth vì đây là model đã train.
# Nếu muốn xóa model thì tự làm riêng, không đưa vào script tự động.

# =========================================================
# 5. CLEAN PYTHON CACHE
# =========================================================
echo
echo "========== CLEAN PYTHON CACHE =========="

find . -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# =========================================================
# 6. CLEAN DOCKER / CONTAINER CACHE
# =========================================================
echo
echo "========== CLEAN DOCKER CACHE =========="

docker container prune -f || true
docker image prune -af || true
docker builder prune -af || true

if [ "$DEEP_CLEAN" = "1" ]; then
    echo "[WARN] DEEP_CLEAN=1: cleaning Docker volumes and containerd unused images"
    docker system prune -af --volumes || true

    if command -v crictl >/dev/null 2>&1; then
        sudo crictl rmi --prune || true
    elif command -v k3s >/dev/null 2>&1; then
        sudo k3s crictl rmi --prune || true
    fi

    sudo journalctl --vacuum-time=2d || true
    sudo apt-get clean || true
    python3 -m pip cache purge 2>/dev/null || true
fi

# =========================================================
# 7. SYNTAX CHECK
# =========================================================
echo
echo "========== PYTHON SYNTAX CHECK =========="

python3 -m py_compile control_loop/main_loop.py
python3 -m py_compile control_loop/rl_client.py
python3 -m py_compile control_loop/state_collector.py
python3 -m py_compile control_loop/controller_client.py
python3 -m py_compile control_loop/metrics.py

python3 -m py_compile rl_engine/state_builder.py
python3 -m py_compile rl_engine/reward.py
python3 -m py_compile rl_engine/offline_env.py
python3 -m py_compile rl_engine/agent/api_serving.py

python3 -m py_compile digital_twin/twin.py
python3 -m py_compile digital_twin/safety.py

python3 -m py_compile experiments/summarize_metrics.py
python3 -m py_compile experiments/run_benchmark_matrix.py 2>/dev/null || true
python3 -m py_compile experiments/plot_comparison.py 2>/dev/null || true

echo "[OK] syntax check passed"

# =========================================================
# 8. BUILD + PUSH IMAGE
# =========================================================
echo
echo "========== BUILD IMAGE =========="

if [ "$NO_CACHE" = "1" ]; then
    docker build --no-cache -t "$IMAGE" .
else
    docker build -t "$IMAGE" .
fi

docker push "$IMAGE"

# =========================================================
# 9. DEPLOY K8S MANIFESTS
# =========================================================
echo
echo "========== APPLY K8S MANIFESTS =========="

kubectl apply -f k3s/base.yaml
kubectl apply -f k3s/serving.yaml

if [ -f k3s/monitoring-jobs.yaml ]; then
    kubectl apply -f k3s/monitoring-jobs.yaml
fi

if [ -f k3s/train-surrogate-job.yaml ]; then
    kubectl apply -f k3s/train-surrogate-job.yaml
fi

# =========================================================
# 10. SET IMAGE
# =========================================================
echo
echo "========== SET DEPLOYMENT IMAGES =========="

if kubectl -n "$NS" get deploy "$CONTROL_DEPLOY" >/dev/null 2>&1; then
    kubectl -n "$NS" set image deployment/"$CONTROL_DEPLOY" "$CONTROL_CONTAINER=$IMAGE"
fi

if kubectl -n "$NS" get deploy "$DQN_DEPLOY" >/dev/null 2>&1; then
    kubectl -n "$NS" set image deployment/"$DQN_DEPLOY" "$DQN_CONTAINER=$IMAGE"
fi

if kubectl -n "$NS" get deploy "$PPO_DEPLOY" >/dev/null 2>&1; then
    kubectl -n "$NS" set image deployment/"$PPO_DEPLOY" "$PPO_CONTAINER=$IMAGE"
fi

# =========================================================
# 11. MOUNT HOST LOGS INTO CONTROL-LOOP
# =========================================================
echo
echo "========== PATCH CONTROL-LOOP LOG VOLUME =========="

LOG_HOST_PATH="$(realpath "$REPO/logs")"

kubectl -n "$NS" patch deployment "$CONTROL_DEPLOY" --type merge -p "{
  \"spec\": {
    \"template\": {
      \"spec\": {
        \"volumes\": [
          {
            \"name\": \"repo-logs\",
            \"hostPath\": {
              \"path\": \"$LOG_HOST_PATH\",
              \"type\": \"DirectoryOrCreate\"
            }
          }
        ],
        \"containers\": [
          {
            \"name\": \"$CONTROL_CONTAINER\",
            \"volumeMounts\": [
              {
                \"name\": \"repo-logs\",
                \"mountPath\": \"/app/logs\"
              }
            ]
          }
        ]
      }
    }
  }
}" || true

# =========================================================
# 12. DEFAULT ENV
# =========================================================
echo
echo "========== SET DEFAULT ENV =========="

kubectl -n "$NS" set env deployment/"$CONTROL_DEPLOY" \
    MODE=collect \
    MODEL_TYPE=dqn \
    EVAL_CONFIG=collect \
    PHASE=warmup \
    ATTACK_TYPE=normal \
    ATTACK_INTENSITY=none \
    RUN_ID=0 \
    ENABLE_GUARD=false \
    ENABLE_TWIN=false \
    ENABLE_LLM=false \
    ACTION_DRY_RUN=true \
    STRICT_RL=true \
    ALLOW_UNTRAINED_FALLBACK=false \
    STATE_DIM=8 \
    SLEEP_TIME=2 \
    RUNTIME_LOG=/app/logs/runtime_eval.csv \
    TRANSITION_LOG=/app/logs/transition_log.csv \
    SURROGATE_MODEL=models/surrogate_model.pkl

# =========================================================
# 13. ROLLOUT RESTART
# =========================================================
echo
echo "========== ROLLOUT RESTART =========="

kubectl -n "$NS" rollout restart deployment/"$CONTROL_DEPLOY" || true
kubectl -n "$NS" rollout restart deployment/"$DQN_DEPLOY" || true
kubectl -n "$NS" rollout restart deployment/"$PPO_DEPLOY" || true

kubectl -n "$NS" rollout status deployment/"$CONTROL_DEPLOY" --timeout=180s || true
kubectl -n "$NS" rollout status deployment/"$DQN_DEPLOY" --timeout=180s || true
kubectl -n "$NS" rollout status deployment/"$PPO_DEPLOY" --timeout=180s || true

echo
echo "========== PODS =========="
kubectl get pods -n "$NS" -o wide

echo
echo "========== DISK AFTER CLEAN/BUILD =========="
df -h || true
docker system df || true

echo
echo "[DONE] Clean + build + deploy completed."
