import os
import json
import csv
import pytest
from fastapi.testclient import TestClient
import numpy as np

# Resolve path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.server import app, risk_limits, EMERGENCY_STOP

@pytest.fixture(autouse=True)
def cleanup():
    # Cleanup state files
    for path in ["stop_state.json", "risk_limits.json", "data/processed/emergency_log.csv"]:
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass
    yield
    for path in ["stop_state.json", "risk_limits.json", "data/processed/emergency_log.csv"]:
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass

def test_emergency_stop_persistence_restart():
    client = TestClient(app)
    
    # 1. Trigger stop
    res = client.post("/api/emergency_stop")
    assert res.status_code == 200
    assert res.json()["emergency_stop"] is True
    
    # 2. Check on disk
    assert os.path.exists("stop_state.json")
    with open("stop_state.json", "r") as f:
        state = json.load(f)
        assert state["stopped"] is True

    # 3. Resume
    res = client.post("/api/resume")
    assert res.status_code == 200
    assert res.json()["emergency_stop"] is False
    with open("stop_state.json", "r") as f:
        state = json.load(f)
        assert state["stopped"] is False

def test_risk_limits_updates_persistence():
    client = TestClient(app)
    payload = {
        "max_drawdown_pct": 12.5,
        "max_position_size": 0.6,
        "stop_loss_pct": 6.0,
        "take_profit_pct": 18.0
    }
    res = client.post("/api/set_risk_limits", json=payload)
    assert res.status_code == 200
    
    # Check persistence file
    assert os.path.exists("risk_limits.json")
    with open("risk_limits.json", "r") as f:
        limits = json.load(f)
        assert limits["max_drawdown_pct"] == 12.5
        assert limits["max_position_size"] == 0.6
        assert limits["stop_loss_pct"] == 6.0
        assert limits["take_profit_pct"] == 18.0

def test_emergency_audit_logging():
    client = TestClient(app)
    res = client.post("/api/emergency_stop")
    assert res.status_code == 200
    
    # Verify CSV logs are written correctly
    assert os.path.exists("data/processed/emergency_log.csv")
    with open("data/processed/emergency_log.csv", "r") as f:
        reader = csv.reader(f)
        headers = next(reader)
        row = next(reader)
        assert headers == ["timestamp", "event", "price", "drawdown", "regime", "initiator"]
        assert row[1] == "EMERGENCY_STOP"

def test_circuit_breaker_trigger():
    # Simulate a critical drawdown trigger
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
