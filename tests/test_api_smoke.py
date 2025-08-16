import os, requests
BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")

def test_health():
    assert requests.get(f"{BASE}/health", timeout=10).json()["status"] == "ok"

def test_ingest():
    r = requests.post(f"{BASE}/ingest", timeout=60)
    assert r.ok

def test_upsert_and_ask_pinecone():
    up = requests.post(f"{BASE}/upsert_pinecone", json={
        "texts": ["Rotate keys with ALTER USER ..."],
        "ids": ["kb-1"],
        "namespace": "demo"
    }, timeout=30).json()
    assert up["status"] == "ok"
    ans = requests.post(f"{BASE}/ask_pinecone", json={
        "question": "How do I rotate Snowflake keys?",
        "top_k": 3,
        "namespace": "demo"
    }, timeout=30).json()
    assert "answer" in ans