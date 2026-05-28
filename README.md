<div align="center">

# ⚡ KinetiMesh

### Federated Kinetic Intelligence Network for Urban Energy Harvesting

[![CI/CD](https://github.com/Unwilling-mcu/kinetimesh/actions/workflows/ci.yml/badge.svg)](https://github.com/Unwilling-mcu/kinetimesh/actions)
[![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Live-green?style=flat-square&logo=fastapi)](https://kinetimesh-api.onrender.com/docs)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-red?style=flat-square&logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-yellow?style=flat-square)](LICENSE)

**Author:** Sanchayan Garai · B.Tech Information Technology · KIIT University  
**GitHub:** [@Unwilling-mcu](https://github.com/Unwilling-mcu)

---

### 🌐 Live Links

| Resource | URL |
|----------|-----|
| 🏠 **Portfolio / Landing Page** | [unwilling-mcu.github.io/kinetimesh](https://unwilling-mcu.github.io/kinetimesh/) |
| ⚡ **Live System Demo (v4)** | [unwilling-mcu.github.io/kinetimesh/KinetiMesh_v4_Dashboard.html](https://unwilling-mcu.github.io/kinetimesh/KinetiMesh_v4_Dashboard.html) |
| 🔌 **Live API (Swagger UI)** | [kinetimesh-api.onrender.com/docs](https://kinetimesh-api.onrender.com/docs) |
| 📊 **API Health Check** | [kinetimesh-api.onrender.com/api/v1/health](https://kinetimesh-api.onrender.com/api/v1/health) |
| 📄 **Research Proposal** | [unwilling-mcu.github.io/kinetimesh/KinetiMesh_Ultimate_Proposal.html](https://unwilling-mcu.github.io/kinetimesh/KinetiMesh_Ultimate_Proposal.html) |
| 📁 **GitHub Repository** | [github.com/Unwilling-mcu/kinetimesh](https://github.com/Unwilling-mcu/kinetimesh) |

> **Note:** The API is on Render's free tier — first request after inactivity may take ~30 seconds to wake up.

</div>

---

## What is KinetiMesh?

KinetiMesh is a full-stack research system that converts **waste mechanical energy** from high-speed rail vibrations, pedestrian footfalls, aerodynamic tunnel drafts, and thermal rail gradients into grid-injectable electricity — governed by a four-layer AI stack.

The core claim: **the energy is already there. The waste is a software problem.**

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  TIER 1 — Physical Sources                                   │
│  Rail PMEH Arrays · Floor PZT Tiles · Wind · Thermal TEG    │
└──────────────────────────┬──────────────────────────────────┘
                           │ AC→DC · Bridge rectifier · LiPo
┌──────────────────────────▼──────────────────────────────────┐
│  TIER 2 — Edge Intelligence (TinyML Mesh)                    │
│  RPi CM4 · TF Lite · LoRaWAN · MQTT/TLS · FedProx client   │
└──────────┬──────────────┬──────────────────┬────────────────┘
           │              │                  │
┌──────────▼───┐  ┌───────▼──────┐  ┌───────▼──────┐
│  FedProx FL  │  │  GraphSAGE   │  │  PPO RL      │
│  LSTM 24h    │  │  Digital     │  │  Power       │
│  Predictor   │  │  Twin        │  │  Router      │
└──────────────┘  └──────────────┘  └──────────────┘
           │              │                  │
┌──────────▼──────────────▼──────────────────▼────────────────┐
│  TIER 4 — City Microgrid + Hyperledger Fabric P2P Market     │
└─────────────────────────────────────────────────────────────┘
```

---

## Core AI Innovations

| # | Innovation | Tech | Novel Claim |
|---|-----------|------|-------------|
| 1 | **FedProx Harvest Prediction** | PyTorch · LSTM · FedProx | First FL across heterogeneous kinetic sources |
| 2 | **GraphSAGE Digital Twin** | PyTorch Geometric · NetworkX | First GNN topology model for kinetic harvesters |
| 3 | **PPO Power Dispatch** | Stable-Baselines3 · Gymnasium | First RL dispatch for multi-source kinetic mesh |
| 4 | **QAOA Energy Routing** | Qiskit · QUBO | First QAOA application to kinetic mesh routing |
| 5 | **P2P Energy Credits** | Hyperledger Fabric · Go | First blockchain market for ambient kinetic energy |

---

## Live API Endpoints

Base URL: `https://kinetimesh-api.onrender.com`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/health` | System health check |
| `GET` | `/api/v1/nodes` | All 18 harvester node states |
| `GET` | `/api/v1/nodes/{id}/power` | Real-time node power output |
| `GET` | `/api/v1/forecast/24h` | 24h LSTM+FedProx forecast |
| `GET` | `/api/v1/twin/topology` | GNN Digital Twin graph state |
| `POST` | `/api/v1/rl/dispatch` | PPO agent power dispatch |
| `GET` | `/api/v1/quantum/route` | QAOA optimal routing result |
| `GET` | `/api/v1/blockchain/ledger` | Energy credit transactions |
| `POST` | `/api/v1/events/train` | Inject train pass event |
| `WS` | `/ws/v1/stream` | Live telemetry WebSocket |

Full interactive docs: [kinetimesh-api.onrender.com/docs](https://kinetimesh-api.onrender.com/docs)

---

## Quick Start

```bash
# Clone
git clone https://github.com/Unwilling-mcu/kinetimesh.git
cd kinetimesh

# Backend (PowerShell)
cd backend
python -m venv venv
.\venv\Scripts\Activate          # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload    # → http://localhost:8000/docs

# FL Training
cd ml/federated
pip install torch numpy networkx
python train_fedprox.py --nodes 18 --rounds 100 --mu 0.01

# RL Training
cd ml/rl
pip install stable-baselines3 gymnasium
python train_ppo.py --timesteps 1000000

# Digital Twin
cd ml/gnn
python digital_twin.py

# Full stack (Docker)
cd infra
docker compose up -d
```

---

## Project Structure

```
KinetiMesh/
├── index.html                        ← Animated landing page
├── KinetiMesh_v4_Dashboard.html      ← Live system demo (10 tabs)
├── KinetiMesh_Ultimate_Proposal.html ← 22-section research proposal
├── README.md
├── .env.example
├── .github/workflows/ci.yml          ← GitHub Actions CI/CD
├── backend/
│   ├── app/main.py                   ← FastAPI + 10 endpoints + WebSocket
│   ├── requirements.txt              ← Server dependencies
│   └── Dockerfile
├── ml/
│   ├── federated/train_fedprox.py    ← FedProx FL ✅ tested
│   ├── gnn/digital_twin.py           ← GraphSAGE DT ✅ tested
│   └── rl/train_ppo.py               ← PPO RL ✅ tested
├── edge/edge_node.py                 ← MicroPython firmware ✅ tested
├── docs/KinetiMesh_IEEE_Paper.tex    ← IEEE LaTeX paper
├── kinetimesh_fl_best.pt             ← Trained FL model weights
└── infra/
    ├── docker-compose.yml            ← 6-service stack
    └── init.sql                      ← TimescaleDB hypertables
```

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Backend** | FastAPI · Uvicorn · WebSocket · Pydantic |
| **ML / AI** | PyTorch · FedProx · Stable-Baselines3 (PPO) · PyTorch Geometric |
| **Database** | TimescaleDB · Redis |
| **Edge** | TensorFlow Lite · LoRaWAN · MQTT · MicroPython |
| **Blockchain** | Hyperledger Fabric · Go chaincode |
| **Quantum** | Qiskit · Qiskit Aer |
| **Infra** | Docker Compose · Kubernetes · GitHub Actions |
| **Frontend** | Three.js · Chart.js · Anime.js |
| **Deployment** | Render (API) · GitHub Pages (frontend) |

---

## Research

**Paper:** *KinetiMesh: A Federated Kinetic Intelligence Network for Multi-Source Urban Ambient Energy Harvesting and AI-Governed Micro-Grid Integration*

LaTeX source: [`docs/KinetiMesh_IEEE_Paper.tex`](docs/KinetiMesh_IEEE_Paper.tex)  
Target venue: IEEE SmartGridComm 2025 / NeurIPS Climate AI Workshop 2025

**Key results (simulation):**
- FedProx achieves MAE = 0.077 kW/node after 100 rounds (vs 0.172 FedAvg)
- GNN Digital Twin prediction error: 4.44% mean across 18 nodes
- PPO agent: +52.1% green utilization vs rule-based baseline
- QAOA routing: ~28% cost reduction vs classical greedy

---

## CI/CD Status

| Check | Status |
|-------|--------|
| Backend Tests (16 tests) | [![Backend](https://github.com/Unwilling-mcu/kinetimesh/actions/workflows/ci.yml/badge.svg)](https://github.com/Unwilling-mcu/kinetimesh/actions) |
| ML Module Tests | [![ML](https://github.com/Unwilling-mcu/kinetimesh/actions/workflows/ci.yml/badge.svg)](https://github.com/Unwilling-mcu/kinetimesh/actions) |
| Docker Build | [![Docker](https://github.com/Unwilling-mcu/kinetimesh/actions/workflows/ci.yml/badge.svg)](https://github.com/Unwilling-mcu/kinetimesh/actions) |

---

## License

Apache 2.0 — Free for research and non-commercial use.

---

<div align="center">

Built by **Sanchayan Garai** · B.Tech Information Technology · KIIT University · 2025

*"The energy is already there. The waste is a software problem."*

</div>