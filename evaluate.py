from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from rag_core import PolicyRAG, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Duke policy RAG reliability.")
    parser.add_argument("--corpus", default="data/corpus.jsonl")
    parser.add_argument("--eval", default="data/eval_questions.jsonl")
    parser.add_argument("--output", default="artifacts/eval_results.json")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--abstain-threshold", type=float, default=0.28)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rag = PolicyRAG(args.corpus)
    questions = load_jsonl(args.eval)
    rows = []
    retrieval_hits = 0
    abstention_correct = 0
    supported_ratios = []
    by_stress = defaultdict(lambda: {"count": 0, "retrieval_hits": 0, "abstention_correct": 0})

    for item in questions:
        result = rag.answer(item["question"], top_k=args.top_k, abstain_threshold=args.abstain_threshold)
        retrieved_doc_ids = [source["doc_id"] for source in result["sources"]]
        expected_doc_id = item.get("expected_doc_id")
        retrieval_hit = expected_doc_id in retrieved_doc_ids if expected_doc_id else result["verdict"] == "abstain"
        abstention_hit = (result["verdict"] == "abstain") == (not item["answerable"])
        retrieval_hits += int(retrieval_hit)
        abstention_correct += int(abstention_hit)
        supported_ratios.append(result["support"]["support_ratio"])

        stress = item["stress_type"]
        by_stress[stress]["count"] += 1
        by_stress[stress]["retrieval_hits"] += int(retrieval_hit)
        by_stress[stress]["abstention_correct"] += int(abstention_hit)

        rows.append(
            {
                "question": item["question"],
                "answerable": item["answerable"],
                "stress_type": stress,
                "expected_doc_id": expected_doc_id,
                "retrieved_doc_ids": retrieved_doc_ids,
                "retrieval_hit": retrieval_hit,
                "abstention_correct": abstention_hit,
                "reliability": result["reliability"],
                "support_ratio": result["support"]["support_ratio"],
                "answer": result["answer"],
            }
        )

    summary = {
        "n": len(questions),
        "retrieval_at_k": retrieval_hits / len(questions),
        "abstention_accuracy": abstention_correct / len(questions),
        "mean_claim_support_ratio": sum(supported_ratios) / len(supported_ratios),
        "by_stress_type": {
            stress: {
                "count": values["count"],
                "retrieval_at_k": values["retrieval_hits"] / values["count"],
                "abstention_accuracy": values["abstention_correct"] / values["count"],
            }
            for stress, values in sorted(by_stress.items())
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
