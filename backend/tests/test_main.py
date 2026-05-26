"""
KinetiMesh Backend Tests
Tests all API endpoints without requiring a live database.
"""
import pytest
from fastapi.testclient import TestClient
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.main import app

client = TestClient(app)

def test_root():
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert data["service"] == "KinetiMesh API"
    assert data["version"] == "3.0.0"

def test_health():
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_get_nodes():
    r = client.get("/api/v1/nodes")
    assert r.status_code == 200
    data = r.json()
    assert "nodes" in data
    assert data["count"] == 18
    assert len(data["nodes"]) == 18
    # Check node structure
    node = data["nodes"][0]
    assert "id" in node
    assert "type" in node
    assert "power_kw" in node
    assert "zone" in node

def test_node_types():
    r = client.get("/api/v1/nodes")
    nodes = r.json()["nodes"]
    types = {n["type"] for n in nodes}
    assert "rail" in types
    assert "floor" in types
    assert "wind" in types
    assert "thermal" in types

def test_get_node_power():
    r = client.get("/api/v1/nodes/R01/power")
    assert r.status_code == 200
    data = r.json()
    assert data["id"] == "R01"
    assert "power_kw" in data
    assert data["power_kw"] > 0

def test_forecast_24h():
    r = client.get("/api/v1/forecast/24h")
    assert r.status_code == 200
    data = r.json()
    assert "forecast" in data
    assert len(data["forecast"]) == 48   # 48 x 30-min slots
    assert data["model"] == "fedprox-lstm-v2"
    assert data["accuracy"] > 0.9

def test_forecast_structure():
    r = client.get("/api/v1/forecast/24h")
    slot = r.json()["forecast"][0]
    assert "hour" in slot
    assert "mean_kw" in slot
    assert "ci_lower" in slot
    assert "ci_upper" in slot
    assert slot["ci_upper"] > slot["ci_lower"]

def test_digital_twin_topology():
    r = client.get("/api/v1/twin/topology")
    assert r.status_code == 200
    data = r.json()
    assert data["nodes"] == 18
    assert data["edges"] == 24
    assert "gnn_error_pct" in data
    assert data["gnn_error_pct"] < 15   # Should be reasonable

def test_rl_dispatch():
    r = client.post("/api/v1/rl/dispatch")
    assert r.status_code == 200
    data = r.json()
    assert "action" in data
    assert len(data["action"]) == 3     # 3 zones
    assert abs(sum(data["action"]) - 1.0) < 0.01   # Must sum to 1
    assert data["green_util_pct"] > 0

def test_quantum_route():
    r = client.get("/api/v1/quantum/route")
    assert r.status_code == 200
    data = r.json()
    assert "advantage_pct" in data
    assert data["advantage_pct"] > 0
    assert data["classical_cost"] == 100
    assert data["quantum_cost"] < 100   # Quantum should be better

def test_blockchain_ledger():
    r = client.get("/api/v1/blockchain/ledger")
    assert r.status_code == 200
    data = r.json()
    assert "transactions" in data
    assert len(data["transactions"]) == 5
    tx = data["transactions"][0]
    assert "hash" in tx
    assert "credits" in tx
    assert tx["credits"] > 0

def test_inject_train_event():
    r = client.post("/api/v1/events/train")
    assert r.status_code == 200
    assert r.json()["status"] == "injected"
    assert r.json()["event"] == "train_pass"

def test_inject_crowd_event():
    r = client.post("/api/v1/events/crowd")
    assert r.status_code == 200
    assert r.json()["status"] == "injected"

def test_total_power_positive():
    r = client.get("/api/v1/nodes")
    assert r.json()["total_kw"] > 0

def test_battery_soc_range():
    r = client.get("/api/v1/nodes")
    for node in r.json()["nodes"]:
        soc = node["battery_soc"]
        assert 0 <= soc <= 100, f"SoC out of range: {soc}"

def test_fl_mae_reasonable():
    r = client.get("/api/v1/nodes")
    for node in r.json()["nodes"]:
        mae = node["fl_mae"]
        assert 0 < mae < 1.0, f"FL MAE unreasonable: {mae}"
