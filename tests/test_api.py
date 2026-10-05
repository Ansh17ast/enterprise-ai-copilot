import pytest
from starlette.testclient import TestClient
from api.main import app, initialize_sample_data
from core.config import settings

@pytest.fixture(scope="module")
def client():
    initialize_sample_data()
    with TestClient(app) as c:
        yield c

def test_health_endpoint(client):
    res = client.get(f"{settings.API_PREFIX}/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["indexed_chunks"] > 0
    assert data["registered_tables"] > 0

def test_schema_endpoint(client):
    res = client.get(f"{settings.API_PREFIX}/analytics/schema")
    assert res.status_code == 200
    data = res.json()
    assert "tables" in data
    assert len(data["tables"]) > 0

def test_query_rag_endpoint(client):
    req = {
        "query": "What is our mandatory retention period for financial transaction logs?",
        "top_k": 3
    }
    res = client.post(f"{settings.API_PREFIX}/query", json=req)
    assert res.status_code == 200
    data = res.json()
    assert data["route_used"] == "unstructured_rag"
    assert len(data["citations"]) > 0
    assert data["execution_time_ms"] > 0
    assert not data["is_refusal"]

def test_query_analytics_endpoint(client):
    req = {
        "query": "What is the total revenue across all regions?",
        "top_k": 3
    }
    res = client.post(f"{settings.API_PREFIX}/query", json=req)
    assert res.status_code == 200
    data = res.json()
    assert data["route_used"] == "structured_analytics"
    assert len(data["citations"]) > 0

def test_query_refusal_guard(client):
    req = {
        "query": "What is our policy for mining asteroids in deep space?",
        "top_k": 3
    }
    res = client.post(f"{settings.API_PREFIX}/query", json=req)
    assert res.status_code == 200
    data = res.json()
    assert data["is_refusal"] is True
    assert "Insufficient context" in data["answer"]
