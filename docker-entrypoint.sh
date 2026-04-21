#!/bin/bash
set -e

echo "========================================"
echo "RL Training Agents - DQN & PPO"
echo "========================================"
echo ""

# Create log directory
mkdir -p /app/logs
mkdir -p /app/results
mkdir -p /app/models
mkdir -p /app/mlruns

# Wait for MLflow to be ready
echo "Waiting for MLflow to be ready..."
for i in {1..30}; do
  if curl -s http://mlflow:5000 > /dev/null 2>&1; then
    echo "MLflow is ready!"
    break
  fi
  echo "Attempt $i/30: Waiting for MLflow..."
  sleep 2
done

echo ""
echo "Starting DQN and PPO training agents..."
echo "View logs at: logs/dqn_agent.log and logs/ppo_agent.log"
echo ""

# Start supervisor to manage both processes
exec /usr/bin/supervisord -c /etc/supervisord.conf