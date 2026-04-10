```markdown
# Autonomous SDN Security System

An intelligent network defense system using **Software-Defined Networking (SDN)** and **Reinforcement Learning (RL)**.

---

## What This Project Demonstrates

- Real-time network traffic monitoring & control  
- AI-driven decision making (DQN / PPO)  
- Automated attack mitigation (DDoS, Packet in Flood, Flow Table Exhaustion, IP spoofing, Porrt Scanning)  
- Digital Twin validation before deployment  
- Full system observability (Prometheus, Grafana, MLflow)  
- Containerized, reproducible environment  

---

## Overview

This project models network defense as a **reinforcement learning problem**:

```

State → RL Agent → Action → SDN Controller → Network → Feedback → Reward

```

The system continuously monitors network conditions and learns optimal policies to defend against attacks.

---

## System Architecture

### High-level

```

Mininet → Controller → RL Agent → Action → Network
↓
Digital Twin
↓
Monitoring

```

---

### Layered Design

```

Training & Evaluation   (experiments/)
↓
RL Agent                (rl_engine/)
↓
Digital Twin            (digital_twin/)
↓
Environment             (env + state_builder + reward)
↓
Controller Client
↓
SDN Controller          (mock Flask / ONOS)
↓
Network (Mininet + attacks)

```

---

## Project Structure

```

autonomous_sdn_security/
├── infra/               # Mock SDN controller + Mininet
├── rl_engine/           # RL agent (DQN / PPO)
├── digital_twin/        # Validation layer
├── mlops/               # MLflow + monitoring
├── experiments/         # Baselines & evaluation
├── traffic_generator/   # Attack simulation
└── docker-compose.yml

````

---

## Tech Stack

| Area | Technology |
|------|-----------|
| RL | PyTorch, Gymnasium |
| Network | Mininet, OpenFlow |
| SDN Controller | ONOS |
| Attack Simulation | hping3, Scapy |
| MLOps | MLflow |
| Monitoring | Prometheus, Grafana |
| Deployment | Docker, K3s |

---

## Getting Started

### 1. Clone repository

```bash
git clone https://github.com/hmm0411/autonomous_sdn_security.git
cd autonomous_sdn_security
````

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Run system

```bash
docker compose up --build
```

---

## How It Works

### Controller

* Flask-based SDN controller mock
* Exposes `/state` API returning network metrics

### RL Agent

* DQN / PPO agents
* Observes 10-dimensional state vector
* Chooses actions:

  * block traffic
  * limit bandwidth
  * redirect
  * isolate

### Digital Twin

* Simulates action effects before deployment
* Prevents unsafe decisions

### Traffic Generator

* Simulates attacks:

  * DDoS
  * Packet in Flood
  * Flow table exhaustion
  * IP spoofing
  * Port Scanning

### MLOps

* MLflow tracks experiments
* Prometheus collects system metrics

---

## Key Design Decisions

* **Mock-first development** → RL training without real SDN dependency
* **Digital Twin safety layer** → validate actions before execution
* **Hybrid reward function**:

  * QoS (latency, packet loss penalties)
  * Security (attack detection accuracy)
  * Stability (penalty for rapid switching)
* **Fully containerized system** → reproducible deployment

---

## Example Outputs

* RL reward convergence
* Reduced network latency under attack
* Detection of abnormal traffic patterns
* Automated mitigation actions

---

## Development Roadmap

### Phase 1

* Mock controller
* Random agent

### Phase 2

* DQN training
* QoS-based reward

### Phase 3

* Digital Twin (ML-based)
* Stability analysis

### Phase 4

* CI/CD pipeline
* Cloud deployment

---

## Notes

This project focuses on **real-world system design**, combining:

* Network infrastructure
* Intelligent decision-making
* Observability and monitoring
* Automation and deployment

---