<div align="center">

# ⚡ KinetiMesh

### Federated Kinetic Intelligence Network for Urban Ambient Energy Harvesting

[![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-red?style=flat-square&logo=pytorch)](https://pytorch.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue?style=flat-square&logo=docker)](https://docker.com)
[![License](https://img.shields.io/badge/License-Apache_2.0-yellow?style=flat-square)](LICENSE)

*The world's first AI-governed, federated kinetic energy harvesting network.*

**Author:** Sanchayan · B.Tech Information Technology  
**GitHub:** [@Unwilling-mcu](https://github.com/Unwilling-mcu)  
**Status:** Research Prototype · v3.0

---

[🚀 Quick Start](#quick-start) · [📐 Architecture](#architecture) · [🧠 AI Stack](#ai-stack) · [📡 API Docs](#api-docs) · [🔬 Research](#research)

</div>

---

## What is KinetiMesh?

KinetiMesh is a full-stack AI platform that converts **waste mechanical energy** from high-speed rail vibrations, pedestrian footfalls, aerodynamic tunnel drafts, and thermal rail gradients into grid-injectable electricity — governed by a four-layer intelligent software stack:

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Edge** | TinyML · LoRaWAN · MQTT | On-device inference, privacy-preserving data collection |
| **FL** | FedProx · LSTM · PyTorch | Federated 24h harvest prediction, no raw data sharing |
| **Digital Twin** | GraphSAGE GNN · NetworkX | Real-time kinetic topology modeling |
| **RL Dispatch** | PPO · Stable-Baselines3 | Demand-aware microgrid power routing |
| **Blockchain** | Hyperledger Fabric | P2P energy credit trading & settlement |
| **Quantum** | QAOA · Qiskit | Quantum-optimized energy routing (research module) |

---

## Quick Start

### Prerequisites
- Docker 24+ and Docker Compose v2
- Python 3.12+ (for local dev)
- Node 20+ (for frontend dev)
- 8GB RAM minimum

### 1. Clone and configure
```bash
git clone https://github.com/Unwilling-mcu/kinetimesh.git
cd kinetimesh
cp .env.example .env
# Edit .env with your configuration
```

### 2. Launch full stack
```bash
docker compose up -d
```

### 3. Access services
| Service | URL | Credentials |
|---------|-----|-------------|
| Dashboard | http://localhost:3000 | — |
| API Docs | http://localhost:8000/docs | — |
| TimescaleDB | localhost:5432 | see .env |
| Grafana | http://localhost:3001 | admin/kinetimesh |
| Prometheus | http://localhost:9090 | — |

### 4. Run FL training
```bash
cd ml/federated
python train_fedprox.py --nodes 18 --rounds 100 --mu 0.01
```

### 5. Run RL agent
```bash
cd ml/rl
python train_ppo.py --timesteps 1000000 --env KinetiMeshGridEnv-v1
```

---

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │           KINETIMESH PLATFORM            │
                    └─────────────────────────────────────────┘

  ┌──────────┐   LoRaWAN   ┌──────────┐   gRPC/REST  ┌──────────────┐
  │ Rail PEH │ ──────────► │  Edge    │ ──────────►  │  FastAPI     │
  │ Floor PEH│             │  Node    │              │  Backend     │
  │ Wind TEG │             │ TinyML   │              │  (Python)    │
  └──────────┘             └──────────┘              └──────┬───────┘
                                                            │
                    ┌───────────────────────────────────────┤
                    │                                       │
              ┌─────▼──────┐   ┌────────────┐   ┌──────────▼──────┐
              │ FedProx FL │   │ GraphSAGE  │   │   PPO RL Agent  │
              │ Aggregator │   │ Digital Twin│   │   Power Router  │
              └─────┬──────┘   └─────┬──────┘   └──────────┬──────┘
                    │                │                      │
              ┌─────▼────────────────▼──────────────────────▼──────┐
              │              TimescaleDB + Redis                    │
              └─────────────────────────┬────────────────────────────┘
                                        │
                    ┌───────────────────▼────────────────────┐
                    │      City Microgrid + Hyperledger       │
                    │      Fabric P2P Energy Credit Market    │
                    └────────────────────────────────────────┘
```

---

## AI Stack

### Federated Learning (FedProx)
- **Model:** LSTM(64,128) → 96-step 24h forecast
- **Algorithm:** FedProx with proximal regularization μ=0.01
- **Privacy:** ε-differential privacy (ε=1.0, δ=1e-5) + SecAgg
- **Target:** MAE < 0.08 kW/node, accuracy > 92%

### Digital Twin (GraphSAGE)
- **Graph:** G=(V,E), |V|=1000–10,000 nodes, |E|=edges weighted by line loss
- **Model:** GraphSAGE (K=3 hops, d=128 hidden dim)
- **Sync:** 60-second live refresh via MQTT → Redis → GNN inference

### RL Dispatch (PPO)
- **Algorithm:** Proximal Policy Optimization (clip ε=0.2, γ=0.99)
- **State:** harvest_vec ⊕ demand_forecast ⊕ battery_soc ⊕ grid_price
- **Action:** Continuous allocation vector ∈ ℝⁿ over n zones
- **Reward:** α·green_util − β·curtailment − γ·grid_import

### Quantum Routing (QAOA) — Research Module
- **Algorithm:** QAOA depth p=3, optimization via COBYLA
- **Problem:** QUBO energy routing formulation
- **Advantage:** O(√N) quantum speedup claim over classical O(N)
- **Backend:** Qiskit Aer simulator (hardware-ready)

---

## API Docs

Full OpenAPI 3.1 spec at `/docs` when running. Key endpoints:

```
GET  /api/v1/nodes                    All harvester node states
GET  /api/v1/nodes/{id}/power         Real-time node power
GET  /api/v1/forecast/24h             24h LSTM+FedProx forecast
GET  /api/v1/twin/topology            GNN Digital Twin graph state
POST /api/v1/rl/dispatch              PPO agent dispatch action
GET  /api/v1/quantum/route            QAOA optimal routing result
GET  /api/v1/blockchain/ledger        Energy credit transactions
WS   /ws/v1/stream                   Live telemetry WebSocket
POST /api/v1/events/train             Inject train pass event
GET  /api/v1/health                   System health check
```

---

## Research

**Paper:** *KinetiMesh: A Federated Kinetic Intelligence Network for Multi-Source Urban Ambient Energy Harvesting* — targeting IEEE Transactions on Smart Grid, 2026.

**Novel contributions:**
1. First FL-based harvest prediction across heterogeneous kinetic sources
2. First GNN Digital Twin of a kinetic energy harvester topology  
3. First RL-governed multi-source kinetic power dispatch
4. First QAOA application to kinetic energy mesh routing
5. First blockchain P2P market for ambient kinetic energy credits

**Key references:**
- Min et al. (AIP Advances, 2024) — Rail PMEH energy densities
- Selim et al. (Energies, 2024) — Pedestrian floor tile harvesting
- Li et al. (NeurIPS, 2020) — FedProx algorithm
- Hamilton et al. (NeurIPS, 2017) — GraphSAGE

---

## License

Apache 2.0 — Free for research and non-commercial use.  
Commercial deployment requires written permission.

---

<div align="center">
Built with 🧠 by Sanchayan · B.Tech IT · 2025
</div>
