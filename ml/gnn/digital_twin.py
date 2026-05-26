"""
KinetiMesh — GraphSAGE GNN Digital Twin
Models the kinetic energy topology of the harvester network.
Author: Sanchayan | B.Tech IT
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import logging

log = logging.getLogger("kinetimesh.gnn")

try:
    from torch_geometric.nn import SAGEConv
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    log.warning("PyTorch Geometric not installed — using fallback GNN")


# ──────────────────────────────────────────
# Node Feature Vector (per harvester node)
# ──────────────────────────────────────────
# x_v = [type_onehot(4), lat, lon, capacity_kw, live_kw, battery_soc, last_event_dt, density]
# dim = 4+2+1+1+1+1+1 = 11

NODE_FEATURE_DIM = 11
HIDDEN_DIM       = 128
N_LAYERS         = 3   # GraphSAGE hops

NODE_TYPES = {"rail": 0, "floor": 1, "wind": 2, "thermal": 3}


class GraphSAGEDigitalTwin(torch.nn.Module):
    """
    GraphSAGE-based Digital Twin for kinetic energy topology.
    Aggregates neighborhood features to predict per-node energy output.

    Aggregation: h_v^(k) = ReLU(W^(k) · MEAN({h_u^(k-1): u ∈ N(v) ∪ {v}}))
    """

    def __init__(self, in_dim: int = NODE_FEATURE_DIM,
                 hidden_dim: int = HIDDEN_DIM, out_dim: int = 1):
        super().__init__()
        if HAS_PYG:
            self.conv1 = SAGEConv(in_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim)
            self.conv3 = SAGEConv(hidden_dim, hidden_dim)
        else:
            # Fallback MLP for environments without PyG
            self.conv1 = torch.nn.Linear(in_dim, hidden_dim)
            self.conv2 = torch.nn.Linear(hidden_dim, hidden_dim)
            self.conv3 = torch.nn.Linear(hidden_dim, hidden_dim)

        self.output = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, out_dim),
            torch.nn.Softplus(),   # Ensure non-negative energy output
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if HAS_PYG:
            h = F.relu(self.conv1(x, edge_index))
            h = F.dropout(h, p=0.2, training=self.training)
            h = F.relu(self.conv2(h, edge_index))
            h = F.relu(self.conv3(h, edge_index))
        else:
            h = F.relu(self.conv1(x))
            h = F.relu(self.conv2(h))
            h = F.relu(self.conv3(h))
        return self.output(h)


@dataclass
class HarvesterNode:
    node_id: str
    node_type: str
    lat: float
    lon: float
    capacity_kw: float
    zone: int
    live_kw: float = 0.0
    battery_soc: float = 0.5
    last_event: float = 0.0
    local_density: float = 0.5

    def to_feature_vector(self) -> List[float]:
        type_onehot = [0.0, 0.0, 0.0, 0.0]
        type_onehot[NODE_TYPES.get(self.node_type, 0)] = 1.0
        return type_onehot + [
            self.lat / 90.0,           # Normalized latitude
            self.lon / 180.0,          # Normalized longitude
            self.capacity_kw / 5.0,    # Normalized capacity
            self.live_kw / 5.0,        # Normalized live output
            self.battery_soc,          # Already [0,1]
            self.last_event,           # Seconds since last event / 3600
            self.local_density,        # Crowd/train density [0,1]
        ]


class KinetiMeshDigitalTwin:
    """
    Full Digital Twin: maintains the graph topology, syncs from MQTT telemetry,
    and runs GNN inference to predict energy flow evolution.
    """

    def __init__(self):
        self.model = GraphSAGEDigitalTwin()
        self.model.eval()
        self.nodes: Dict[str, HarvesterNode] = {}
        self.edges: List[tuple] = []   # (from_id, to_id, weight)
        self._build_default_topology()
        log.info(f"Digital Twin initialized: {len(self.nodes)} nodes, {len(self.edges)} edges")

    def _build_default_topology(self):
        """Build KinetiMesh default 18-node topology."""
        configs = [
            ("R01","rail",22.57,88.36,1.8,0),("R02","rail",22.571,88.361,2.1,0),
            ("R03","rail",22.572,88.362,1.6,0),("R04","rail",22.573,88.363,1.9,0),
            ("R05","rail",22.574,88.364,1.4,0),("R06","rail",22.575,88.365,2.3,0),
            ("F01","floor",22.58,88.37,0.24,1),("F02","floor",22.581,88.371,0.18,1),
            ("F03","floor",22.582,88.372,0.22,1),("F04","floor",22.583,88.373,0.19,1),
            ("F05","floor",22.584,88.374,0.21,1),("F06","floor",22.585,88.375,0.17,1),
            ("W01","wind",22.57,88.38,0.15,2),("W02","wind",22.571,88.381,0.12,2),
            ("T01","thermal",22.57,88.36,0.08,0),("T02","thermal",22.575,88.365,0.07,0),
            ("S01","floor",22.59,88.39,0.09,2),("S02","floor",22.591,88.391,0.08,2),
        ]
        for nid,nt,lat,lon,cap,zone in configs:
            self.nodes[nid] = HarvesterNode(nid,nt,lat,lon,cap,zone)
        # Edges: intra-zone connections + zone-to-aggregator
        self.edges = [(f"R0{i}",f"R0{i+1}",0.95) for i in range(1,6)]
        self.edges += [(f"F0{i}",f"F0{i+1}",0.92) for i in range(1,6)]
        self.edges += [("W01","W02",0.9),("T01","T02",0.88)]

    def sync_from_mqtt(self, telemetry: dict):
        """Update node states from live MQTT telemetry packet."""
        for nid, data in telemetry.items():
            if nid in self.nodes:
                self.nodes[nid].live_kw      = data.get("power_kw", 0)
                self.nodes[nid].battery_soc  = data.get("battery_soc", 0.5)
                self.nodes[nid].last_event   = data.get("last_event", 0)
                self.nodes[nid].local_density = data.get("density", 0.5)

    def build_graph(self):
        """Convert topology to PyTorch tensors for GNN inference."""
        node_ids = list(self.nodes.keys())
        id2idx   = {nid: i for i, nid in enumerate(node_ids)}
        x = torch.tensor([self.nodes[nid].to_feature_vector() for nid in node_ids],
                          dtype=torch.float32)
        edges_fwd = [(id2idx[u], id2idx[v]) for u,v,_ in self.edges if u in id2idx and v in id2idx]
        edges_rev = [(v,u) for u,v in edges_fwd]
        all_edges = edges_fwd + edges_rev
        if all_edges:
            edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2,0), dtype=torch.long)
        return x, edge_index, node_ids

    def infer(self) -> Dict[str, float]:
        """Run GNN inference → predicted energy output per node."""
        x, edge_index, node_ids = self.build_graph()
        with torch.no_grad():
            pred = self.model(x, edge_index)
        return {nid: float(pred[i].item()) for i, nid in enumerate(node_ids)}

    def get_state_summary(self) -> dict:
        predictions = self.infer()
        total_live = sum(n.live_kw for n in self.nodes.values())
        total_pred = sum(predictions.values())
        error_pct  = abs(total_live - total_pred) / max(total_live, 0.01) * 100
        return {
            "n_nodes": len(self.nodes),
            "n_edges": len(self.edges),
            "total_live_kw": round(total_live, 4),
            "total_predicted_kw": round(total_pred, 4),
            "error_pct": round(error_pct, 2),
            "node_predictions": {k: round(v, 4) for k,v in predictions.items()},
        }


if __name__ == "__main__":
    twin = KinetiMeshDigitalTwin()
    # Simulate MQTT telemetry
    fake_telemetry = {f"R0{i}": {"power_kw": 1.5+i*0.1, "battery_soc": 0.6} for i in range(1,7)}
    twin.sync_from_mqtt(fake_telemetry)
    summary = twin.get_state_summary()
    print(f"Digital Twin state: {len(summary['node_predictions'])} nodes")
    print(f"Total live: {summary['total_live_kw']} kW")
    print(f"Total predicted: {summary['total_predicted_kw']} kW")
    print(f"GNN error: {summary['error_pct']}%")
