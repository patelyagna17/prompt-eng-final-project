# app/eval_ragas.py
import json
import argparse
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from app.rag_pipeline import RagService


def load_gold(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def build_dataset(svc: RagService, gold):
    qs, ans, ctxs, gt = [], [], [], []
    for item in gold:
        q = item["question"]
        k = item.get("k", 6)
        res = svc.answer(q, k=k)
        docs = svc._retrieve(q, k=k)  # contexts used by RAGAS
        qs.append(q)
        ans.append(res["answer"])
        ctxs.append([d.page_content for d in docs])
        gt.append(item.get("ground_truth", ""))  # optional
    return Dataset.from_dict(
        {"question": qs, "answer": ans, "contexts": ctxs, "ground_truth": gt}
    )


def main(gold_path: str):
    svc = RagService()
    svc.ensure_kb()
    gold = load_gold(gold_path)
    ds = build_dataset(svc, gold)
    result = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision])
    print(result)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="tests/fixtures/gold.jsonl")
    args = ap.parse_args()
    main(args.gold)
