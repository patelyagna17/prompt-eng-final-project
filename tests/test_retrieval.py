# tests/test_retrieval.py
import os
import pytest
from app.rag_pipeline import RagService


def kb_empty():
    return not os.path.isdir("data/raw") or len(os.listdir("data/raw")) == 0


@pytest.mark.skipif(kb_empty(), reason="No docs in data/raw; skipping.")
def test_retrieval_returns_docs():
    svc = RagService()
    svc.ensure_kb()
    docs = svc._retrieve("test", k=3)
    assert isinstance(docs, list)
    assert len(docs) > 0
