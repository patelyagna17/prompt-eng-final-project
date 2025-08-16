import requests

BASE = "http://127.0.0.1:8000"

def test_health():
    r = requests.get(f"{BASE}/health", timeout=5)
    assert r.ok and r.json().get("status") == "ok"

def test_config():
    r = requests.get(f"{BASE}/config", timeout=5)
    assert r.ok and {"llm_backend","embeddings_backend","has_key"} <= r.json().keys()