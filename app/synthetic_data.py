# app/synthetic_data.py
import json
import random
import argparse
from app.utils import SETTINGS
from app.loaders import load_docs
from app.chunkers import split_docs

PROMPT = """You are creating training/eval data.
Given the snippet, produce:
- question: a realistic user question answerable from it.
- ground_truth: a faithful 2–5 sentence answer grounded in the snippet.
Return compact JSON with keys: question, ground_truth.
SNIPPET:
{snippet}
"""


def via_openai(snippet: str):
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    j = llm.invoke(PROMPT.format(snippet=snippet)).content
    return json.loads(j)


def fallback(snippet: str):
    # simple deterministic pair if no API key/model available
    q = "Summarize the key point from the provided documentation snippet."
    gt = snippet[:500]
    return {"question": q, "ground_truth": gt}


def make_pairs(n: int, out_path: str):
    docs = load_docs(SETTINGS.raw_dir)
    chunks = split_docs(docs)
    if not chunks:
        raise SystemExit("No docs in data/raw. Add files, then rerun.")
    use_openai = bool(SETTINGS.openai_api_key)
    pairs = []
    for c in random.sample(chunks, min(n, len(chunks))):
        snip = c.page_content[:1200]
        try:
            pairs.append(via_openai(snip) if use_openai else fallback(snip))
        except Exception:
            pairs.append(fallback(snip))
    with open(out_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add
