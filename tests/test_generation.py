# tests/test_generation.py
import os
import pytest
from app.rag_pipeline import RagService


def kb_empty():
    return not os.path.isdir("data/raw") or len(os.listdir("data/raw")) == 0


@pytest.mark.skipif(kb_empty(), reason="No docs in data/raw; skipping.")
def test_answer_has_sources_or_guardrail():
    svc = RagService()
    svc.ensure_kb()
    res = svc.answer("What does the documentation cover?", k=3)
    txt = res["answer"].lower()
    assert ("sources:" in txt) or ("insufficient context" in txt)
