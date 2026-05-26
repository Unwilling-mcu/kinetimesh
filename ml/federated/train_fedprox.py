"""
KinetiMesh — FedProx Federated Learning Training Script
Algorithm: FedProx (Li et al., NeurIPS 2020)
Model: LSTM-based 24h energy harvest forecaster
Author: Sanchayan | B.Tech IT
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fedprox")


# ──────────────────────────────────────────
# 1. LSTM Harvest Forecasting Model
# ──────────────────────────────────────────

class HarvestLSTM(nn.Module):
    """
    LSTM model for 24h energy harvest forecasting.
    Input:  [vibration_amplitude, frequency_centroid, temperature, schedule_flag]
    Output: 96 time steps (15-min intervals = 24h)
    """
    def __init__(self, input_dim: int = 4, hidden_dim: int = 64,
                 num_layers: int = 2, output_steps: int = 96):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_steps),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # Last timestep → 96-step forecast


# ──────────────────────────────────────────
# 2. Simulated Edge Node Client
# ──────────────────────────────────────────

@dataclass
class EdgeNodeClient:
    """Simulates a KinetiMesh edge harvester node running local FL training."""
    node_id: str
    node_type: str          # "rail" | "floor" | "wind" | "thermal"
    base_power: float       # Baseline kW output
    local_epochs: int = 5
    batch_size: int = 32
    lr: float = 1e-3

    def generate_local_data(self, n_samples: int = 500):
        """Generate synthetic vibration/footfall time-series for this node type."""
        t = np.linspace(0, 4*np.pi, n_samples)
        if self.node_type == "rail":
            # Rail: high frequency (30-650 Hz range), millimeter amplitudes
            amp   = self.base_power * (0.8 + 0.4*np.sin(t*3) + 0.1*np.random.randn(n_samples))
            freq  = 120 + 80*np.sin(t*2) + 10*np.random.randn(n_samples)  # Hz centroid
            temp  = 35 + 5*np.sin(t*0.5) + np.random.randn(n_samples)
            sched = (np.sin(t*4) > 0.6).astype(float)  # Train schedule
        elif self.node_type == "floor":
            # Floor: low frequency (1-3 Hz), crowd-density driven
            amp   = self.base_power * (0.5 + 0.8*abs(np.sin(t*0.8)) + 0.05*np.random.randn(n_samples))
            freq  = 1.5 + 1.0*abs(np.sin(t)) + 0.2*np.random.randn(n_samples)
            temp  = 28 + 3*np.sin(t*0.3) + np.random.randn(n_samples)
            sched = (np.sin(t*2) > 0.3).astype(float)
        else:
            amp   = self.base_power * (0.6 + 0.4*np.random.rand(n_samples))
            freq  = 10 + 5*np.random.rand(n_samples)
            temp  = 30 + np.random.randn(n_samples)
            sched = np.zeros(n_samples)

        X = np.column_stack([amp, freq/1000, temp/100, sched]).astype(np.float32)
        y = (amp + 0.1*np.random.randn(n_samples)).astype(np.float32)
        return torch.tensor(X), torch.tensor(y)

    def local_train(self, global_model: HarvestLSTM, mu: float) -> dict:
        """
        FedProx local training with proximal term.
        Objective: min F_k(w) + (mu/2)||w - w_global||^2
        """
        local_model = deepcopy(global_model)
        optimizer = torch.optim.Adam(local_model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        X, y = self.generate_local_data()
        # Reshape for LSTM: (batch, seq_len, features)
        X = X.unsqueeze(1).expand(-1, 8, -1)  # 8-step input window

        dataset = torch.utils.data.TensorDataset(X, y)
        loader  = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        local_model.train()
        total_loss = 0.0

        for epoch in range(self.local_epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = local_model(xb)
                task_loss = criterion(pred[:, 0], yb)

                # FedProx proximal term: penalize drift from global model
                prox_term = 0.0
                for w_local, w_global in zip(local_model.parameters(), global_model.parameters()):
                    prox_term += ((w_local - w_global.detach()) ** 2).sum()

                loss = task_loss + (mu / 2) * prox_term
                loss.backward()
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

        # Return model weight delta (gradient) — NOT raw data
        delta = {k: (local_model.state_dict()[k] - global_model.state_dict()[k])
                 for k in global_model.state_dict()}

        avg_loss = total_loss / (len(loader) * self.local_epochs)
        mae = float(avg_loss ** 0.5)
        log.info(f"  Node {self.node_id} ({self.node_type}) — local MAE: {mae:.4f}")
        return {"delta": delta, "n_samples": len(X), "mae": mae, "node_id": self.node_id}


# ──────────────────────────────────────────
# 3. FedProx Aggregation Server
# ──────────────────────────────────────────

class FedProxServer:
    """
    Federated aggregation server.
    Implements FedProx weighted averaging across heterogeneous clients.
    """
    def __init__(self, model: HarvestLSTM):
        self.global_model = model
        self.round_history: List[dict] = []

    def aggregate(self, client_results: List[dict]) -> float:
        """Weighted FedAvg aggregation of client weight deltas."""
        total_samples = sum(r["n_samples"] for r in client_results)
        new_state = deepcopy(self.global_model.state_dict())

        # Zero out the state to accumulate weighted updates
        for k in new_state:
            new_state[k] = torch.zeros_like(new_state[k])

        for result in client_results:
            weight = result["n_samples"] / total_samples
            for k, delta in result["delta"].items():
                new_state[k] += weight * delta

        # Apply aggregated delta to global model
        for k in new_state:
            self.global_model.state_dict()[k] += new_state[k]

        avg_mae = sum(r["mae"] * r["n_samples"] for r in client_results) / total_samples
        return avg_mae

    def run_round(self, clients: List[EdgeNodeClient], mu: float) -> dict:
        """Execute one FL communication round."""
        results = [c.local_train(self.global_model, mu=mu) for c in clients]
        global_mae = self.aggregate(results)
        round_stats = {
            "round": len(self.round_history) + 1,
            "global_mae": global_mae,
            "n_clients": len(clients),
            "per_node": {r["node_id"]: r["mae"] for r in results},
        }
        self.round_history.append(round_stats)
        return round_stats


# ──────────────────────────────────────────
# 4. Main Training Loop
# ──────────────────────────────────────────

def train(args):
    log.info(f"KinetiMesh FedProx Training | nodes={args.nodes} rounds={args.rounds} mu={args.mu}")

    # Initialize global model
    global_model = HarvestLSTM(input_dim=4, hidden_dim=64, num_layers=2, output_steps=96)
    server = FedProxServer(global_model)

    # Create simulated edge node clients
    node_configs = [
        ("R01","rail",1.8),("R02","rail",2.1),("R03","rail",1.6),
        ("R04","rail",1.9),("R05","rail",1.4),("R06","rail",2.3),
        ("F01","floor",0.24),("F02","floor",0.18),("F03","floor",0.22),
        ("F04","floor",0.19),("F05","floor",0.21),("F06","floor",0.17),
        ("W01","wind",0.15),("W02","wind",0.12),
        ("T01","thermal",0.08),("T02","thermal",0.07),
        ("S01","floor",0.09),("S02","floor",0.08),
    ][:args.nodes]

    clients = [EdgeNodeClient(nid, ntype, base, local_epochs=args.local_epochs)
               for nid, ntype, base in node_configs]

    best_mae = float("inf")
    for r in range(1, args.rounds + 1):
        # Randomly select fraction of clients per round (C=0.8)
        selected = np.random.choice(clients, size=max(1, int(0.8*len(clients))), replace=False)
        stats = server.run_round(list(selected), mu=args.mu)

        log.info(f"Round {r:3d}/{args.rounds} | Global MAE: {stats['global_mae']:.4f} "
                 f"| Clients: {stats['n_clients']}")

        if stats["global_mae"] < best_mae:
            best_mae = stats["global_mae"]
            torch.save(global_model.state_dict(), "kinetimesh_fl_best.pt")

        if best_mae < 0.08:
            log.info(f"Target MAE reached at round {r}! Best MAE: {best_mae:.4f}")
            break

    log.info(f"Training complete. Best global MAE: {best_mae:.4f}")
    log.info(f"Model saved to kinetimesh_fl_best.pt")
    return best_mae


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KinetiMesh FedProx Training")
    parser.add_argument("--nodes",        type=int,   default=18,   help="Number of edge nodes")
    parser.add_argument("--rounds",       type=int,   default=100,  help="FL communication rounds")
    parser.add_argument("--mu",           type=float, default=0.01, help="FedProx proximal mu")
    parser.add_argument("--local-epochs", type=int,   default=5,    help="Local epochs per round")
    args = parser.parse_args()
    train(args)
