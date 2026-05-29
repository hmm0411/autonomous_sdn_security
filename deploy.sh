#!/bin/bash
set -e

NAMESPACE="sdn-security"

echo "=================================================="
echo "[CD] NT548 SDN MLOps Local Deployment Pipeline"
echo "Namespace: $NAMESPACE"
echo "=================================================="

echo ""
echo "[0/9] Check Kubernetes connection..."
kubectl get nodes
kubectl get ns $NAMESPACE || kubectl create ns $NAMESPACE

echo ""
echo "[1/9] Validate Python files..."
python3 -m py_compile realtime_pipeline.py
python3 -m py_compile mlops/main.py
python3 -m py_compile control_loop/metrics.py
python3 -m py_compile control_loop/main_loop.py
python3 -m py_compile control_loop/rl_client.py
python3 -m py_compile rl_engine/agent/api_serving.py
python3 -m py_compile rl_engine/agent/train_dqn.py
python3 -m py_compile rl_engine/agent/train_ppo.py
python3 -m py_compile traffic_generator/validate_attack_scenarios.py

echo ""
echo "[2/9] Validate Grafana dashboard JSON..."
python3 -m json.tool grafana/dashboards/sdn_rl_dashboard.json > /tmp/dashboard_check.json
echo "Grafana dashboard JSON OK"

echo ""
echo "[3/9] Validate Kubernetes YAML..."
python3 - <<'PY'
import yaml

files = [
    "k8s/base.yaml",
    "k8s/serving.yaml",
    "k8s/monitoring-jobs.yaml",
]

for path in files:
    with open(path, "r", encoding="utf-8") as f:
        list(yaml.safe_load_all(f))
    print(f"YAML OK: {path}")
PY

echo ""
echo "[4/9] Check important metric names..."
grep -R "sdn_current_score" realtime_pipeline.py grafana/dashboards/sdn_rl_dashboard.json
grep -R "sdn_rl_action" realtime_pipeline.py grafana/dashboards/sdn_rl_dashboard.json
grep -R "sdn_packet_rate" realtime_pipeline.py grafana/dashboards/sdn_rl_dashboard.json
grep -R "sdn_flow_count" realtime_pipeline.py grafana/dashboards/sdn_rl_dashboard.json
grep -R "alerts_received_total" mlops/main.py

echo ""
echo "[5/9] Apply Kubernetes manifests..."
kubectl apply -f k8s/base.yaml
kubectl apply -f k8s/serving.yaml
kubectl apply -f k8s/monitoring-jobs.yaml

echo ""
echo "[6/9] Update Grafana dashboard ConfigMap..."
kubectl delete configmap grafana-dashboards -n $NAMESPACE --ignore-not-found

kubectl create configmap grafana-dashboards \
  -n $NAMESPACE \
  --from-file=sdn_rl_dashboard.json=grafana/dashboards/sdn_rl_dashboard.json

echo ""
echo "[7/9] Restart deployments..."
for deploy in \
  prometheus \
  grafana \
  alertmanager \
  mlflow \
  rl-serving-dqn \
  rl-serving-ppo \
  control-loop \
  sdn-defense-agent \
  auto-trigger
do
  echo "Restarting deployment/$deploy ..."
  kubectl rollout restart deployment/$deploy -n $NAMESPACE || true
done

echo ""
echo "[8/9] Wait for rollout..."
for deploy in \
  prometheus \
  grafana \
  alertmanager \
  mlflow \
  rl-serving-dqn \
  rl-serving-ppo \
  control-loop \
  sdn-defense-agent \
  auto-trigger
do
  echo "Waiting for deployment/$deploy ..."
  kubectl rollout status deployment/$deploy -n $NAMESPACE --timeout=240s || true
done

echo ""
echo "[9/9] Show final status..."
echo ""
echo "Pods:"
kubectl get pods -n $NAMESPACE

echo ""
echo "Services:"
kubectl get svc -n $NAMESPACE

echo ""
echo "Jobs:"
kubectl get jobs -n $NAMESPACE

echo ""
echo "CronJobs:"
kubectl get cronjob -n $NAMESPACE

echo ""
echo "=================================================="
echo "[CD] Deployment completed."
echo "=================================================="