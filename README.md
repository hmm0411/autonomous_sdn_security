# Autonomous SDN Security (RL + SDN + MLOps)

## What this repository is

This project builds an **autonomous defense loop for Software-Defined Networking (SDN)**.
It uses reinforcement learning to choose mitigation actions from live network signals, then evaluates and serves those models with MLOps tooling.

## Main purpose

The repository is intended to answer this practical question:

**Can an RL agent detect abnormal SDN traffic patterns and choose useful mitigation actions automatically, with measurable QoS/security impact?**

It supports that goal by providing:
- offline training pipelines (DQN and PPO),
- runtime control loop integration,
- attack traffic generation in Mininet,
- experiment/evaluation scripts,
- model serving and monitoring components.

## What output you get from this project

When running this repository, typical outputs are:

1. **Model files**
   - `models/dqn_model.pth`
   - `models/ppo_model.pth`

2. **Evaluation artifacts**
   - CSV reports under `results/`
   - Example: `results/evaluation/offline_eval_results.csv`

3. **Runtime and training logs**
   - `logs/`, `runs/`, `mlruns/`

4. **Inference outputs (API JSON)**
   - `/health`, `/predict`, `/reload` from `rl_engine.agent.api_serving`

5. **Prometheus metrics**
   - training/serving/control-loop metrics for monitoring dashboards

## High-level architecture

```text
Traffic/Telemetry -> State Builder -> RL Model (DQN/PPO) -> Action
        ^                                                |
        |                                                v
   Controller Metrics <- Control Loop <- SDN Controller Execution
```

## Repository layout

- `rl_engine/` - RL configs, environments, agents, training, serving API
- `control_loop/` - runtime loop, state collection, action execution
- `traffic_generator/` - Mininet topology + attack scenario generation
- `experiments/` - benchmarking and offline evaluation scripts
- `digital_twin/` - transition collection and twin/safety-related utilities
- `mlops/` - alert webhook, retrain trigger, promotion helpers
- `k3s/`, `prometheus/`, `grafana/`, `alertmanager/` - deployment/observability configs

---

## How to clone

```bash
git clone https://github.com/hmm0411/autonomous_sdn_security.git
cd autonomous_sdn_security
```

## Local setup

Use Python 3.10+:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Create working directories:

```bash
mkdir -p data/processed models results logs mlruns runs
```

---

## How to run (core workflows)

### A) Train models (offline)

Requirement: `data/processed/train_data.csv`

```bash
python -m rl_engine.agent.train_dqn
python -m rl_engine.agent.train_ppo
```

Expected result:
- trained checkpoints in `models/`
- run artifacts in `runs/`, `mlruns/`, `logs/`

### B) Start serving APIs for inference

DQN service:
```bash
python -m rl_engine.agent.api_serving --port 8000
```

PPO service:
```bash
MODEL_TYPE=ppo python -m rl_engine.agent.api_serving --port 8001
```

Example `/predict` response:

```json
{
  "action": 2,
  "action_staging": 2,
  "model": "dqn",
  "latency_seconds": 0.002
}
```

### C) Run runtime control loop

```bash
MODE=rl MODEL_TYPE=dqn ACTION_DRY_RUN=true python -m control_loop.main_loop
```

`ACTION_DRY_RUN=true` is recommended for safe testing.

### D) Run offline evaluation

Requirements:
- `data/processed/val_data.csv`
- `data/processed/test_data.csv`

```bash
python -m experiments.evaluate_offline
```

Expected result:
- offline comparison metrics in `results/evaluation/`

### E) Run Mininet attack simulation (optional)

Requires Mininet + Open vSwitch + reachable SDN controller.

```bash
sudo python -m traffic_generator.run
```

---

## Action IDs used by agents/control loop

- `0` = `no_action`
- `1` = `block_suspicious_flow`
- `2` = `limit_bandwidth`
- `3` = `redirect_traffic`
- `4` = `isolate_device`

## Important notes

- This branch has Dockerfiles and K3s manifests, but **no root `docker-compose.yml`**.
- For experiments, run in dry-run mode first, then enable real enforcement only when controller mappings are verified.
