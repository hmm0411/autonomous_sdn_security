# Autonomous SDN Security

Autonomous SDN Security is a research and engineering project for **AI-driven network defense** in Software-Defined Networking (SDN). It combines reinforcement learning (DQN/PPO), runtime control logic, attack traffic simulation, and MLOps components for model lifecycle and monitoring.

## Purpose of this repository

This repository is built to:
- train RL agents that map SDN telemetry to mitigation actions,
- evaluate those agents against attack scenarios,
- serve trained models as inference APIs,
- run a control loop that applies safe actions to the SDN controller,
- support observability and retraining/promotion workflows.

## What the project outputs

Depending on what you run, the project produces:
- **Trained models**: `models/dqn_model.pth`, `models/ppo_model.pth`
- **Training logs/artifacts**: `runs/`, `mlruns/`, `logs/`
- **Evaluation reports**: CSV files under `results/` (for example `results/evaluation/offline_eval_results.csv`)
- **Serving/API output**: JSON responses from `/health`, `/predict`, `/reload`
- **Runtime metrics**: Prometheus-compatible metrics from training/serving/control-loop processes

## Repository structure (current)

- `rl_engine/` - RL environments, agents, training, serving
- `control_loop/` - runtime decision loop and controller execution layer
- `digital_twin/` - surrogate/twin utilities and transition logging
- `traffic_generator/` - Mininet topology and attack generation
- `experiments/` - offline evaluation and scenario benchmarking scripts
- `mlops/` - alert webhook, retrain trigger, and promotion orchestration
- `k3s/`, `prometheus/`, `grafana/`, `alertmanager/` - deployment and observability configs

## How to clone

```bash
git clone https://github.com/hmm0411/autonomous_sdn_security.git
cd autonomous_sdn_security
```

## Setup

Use Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Create runtime folders:

```bash
mkdir -p data/processed models results logs mlruns runs
```

## How to run

### 1) Train RL models (offline)

The training scripts expect `data/processed/train_data.csv`.

```bash
python -m rl_engine.agent.train_dqn
python -m rl_engine.agent.train_ppo
```

### 2) Run model serving API

```bash
# DQN serving (default MODEL_TYPE=dqn)
python -m rl_engine.agent.api_serving --port 8000

# PPO serving
MODEL_TYPE=ppo python -m rl_engine.agent.api_serving --port 8001
```

### 3) Run the runtime control loop

```bash
MODE=rl MODEL_TYPE=dqn ACTION_DRY_RUN=true python -m control_loop.main_loop
```

### 4) Run offline evaluation

Requires evaluation datasets (for example `data/processed/val_data.csv` and `data/processed/test_data.csv`).

```bash
python -m experiments.evaluate_offline
```

### 5) Run Mininet attack simulation (optional, system-level)

Requires Mininet/Open vSwitch and an SDN controller endpoint.

```bash
sudo python -m traffic_generator.run
```

## Typical API output

Example `/predict` response:

```json
{
  "action": 2,
  "action_staging": 2,
  "model": "dqn",
  "latency_seconds": 0.002
}
```

## Action semantics used by the system

- `0`: no_action
- `1`: block_suspicious_flow
- `2`: limit_bandwidth
- `3`: redirect_traffic
- `4`: isolate_device

## Notes

- This branch currently provides Dockerfiles and K3s manifests, but no root `docker-compose.yml`.
- For safe experiments, keep `ACTION_DRY_RUN=true` in the control loop unless you explicitly want to enforce controller actions.
