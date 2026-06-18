from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import gradio as gr

from rag_core import PolicyRAG, load_jsonl


EXAMPLES = [
    "When is the renewal aid application due for returning students?",
    "if my aid app is late am i cooked / does eligibility go away",
    "Can I take Duke financial aid with me when studying away?",
    "What are the three commitments in the Duke Community Standard?",
    "What is the exact penalty for a first plagiarism violation?",
]


@lru_cache(maxsize=1)
def get_rag() -> PolicyRAG:
    return PolicyRAG("data/corpus.jsonl")


def ask_policy_bot(question: str, top_k: int, threshold: float):
    if not question.strip():
        return "Ask a policy question first.", "Not run", "[]", "[]"
    result = get_rag().answer(question.strip(), top_k=int(top_k), abstain_threshold=float(threshold))
    source_rows = [
        [
            idx + 1,
            source["title"],
            source["doc_id"],
            round(source["score"], 3),
            source["url"],
            source["text"],
        ]
        for idx, source in enumerate(result["sources"])
    ]
    claim_rows = [
        [
            idx + 1,
            claim["claim"],
            "supported" if claim["supported"] else "weak / unsupported",
            claim["term_overlap"],
        ]
        for idx, claim in enumerate(result["support"]["claims"])
    ]
    reliability = (
        f"{result['reliability']} | verdict={result['verdict']} | "
        f"top retrieval score={result['top_score']:.3f} | "
        f"claim support={result['support']['support_ratio']:.2f}"
    )
    return result["answer"], reliability, source_rows, claim_rows


def run_eval(top_k: int, threshold: float):
    rag = get_rag()
    questions = load_jsonl("data/eval_questions.jsonl")
    rows = []
    retrieval_hits = 0
    abstention_hits = 0
    support_ratios = []
    stress_counts = {}

    for item in questions:
        result = rag.answer(item["question"], top_k=int(top_k), abstain_threshold=float(threshold))
        retrieved_doc_ids = [source["doc_id"] for source in result["sources"]]
        expected_doc_id = item.get("expected_doc_id")
        retrieval_hit = expected_doc_id in retrieved_doc_ids if expected_doc_id else result["verdict"] == "abstain"
        abstention_hit = (result["verdict"] == "abstain") == (not item["answerable"])
        retrieval_hits += int(retrieval_hit)
        abstention_hits += int(abstention_hit)
        support_ratios.append(result["support"]["support_ratio"])

        stress = item["stress_type"]
        stress_counts.setdefault(stress, {"count": 0, "retrieval": 0, "abstention": 0})
        stress_counts[stress]["count"] += 1
        stress_counts[stress]["retrieval"] += int(retrieval_hit)
        stress_counts[stress]["abstention"] += int(abstention_hit)

        rows.append(
            [
                item["question"],
                stress,
                "yes" if item["answerable"] else "no",
                expected_doc_id or "none",
                ", ".join(retrieved_doc_ids),
                "hit" if retrieval_hit else "miss",
                "correct" if abstention_hit else "wrong",
                result["reliability"],
                round(result["support"]["support_ratio"], 2),
            ]
        )

    summary = {
        "retrieval@k": round(retrieval_hits / len(questions), 3),
        "abstention_accuracy": round(abstention_hits / len(questions), 3),
        "mean_claim_support": round(sum(support_ratios) / len(support_ratios), 3),
        "stress_breakdown": {
            stress: {
                "count": values["count"],
                "retrieval@k": round(values["retrieval"] / values["count"], 3),
                "abstention_accuracy": round(values["abstention"] / values["count"], 3),
            }
            for stress, values in sorted(stress_counts.items())
        },
    }
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/latest_eval.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    return json.dumps(summary, indent=2), rows


with gr.Blocks(title="Duke Policy RAG Reliability Checker", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Duke Policy RAG Reliability Checker

        A mini NLP reliability prototype for testing whether a RAG assistant can answer Duke student policy questions
        with grounded citations. The goal is not just to answer, but to show when the answer is supported, weak, or should abstain.
        """
    )

    with gr.Tab("Ask"):
        with gr.Row():
            with gr.Column(scale=2):
                question = gr.Textbox(label="Policy question", lines=3, placeholder="Ask about aid renewal, summer aid, study away, or conduct.")
                with gr.Row():
                    top_k = gr.Slider(1, 5, value=3, step=1, label="Retrieved sources")
                    threshold = gr.Slider(0.0, 0.8, value=0.28, step=0.01, label="Abstention threshold")
                ask = gr.Button("Check Answer", variant="primary")
                gr.Examples(EXAMPLES, inputs=question)
            with gr.Column(scale=3):
                answer = gr.Textbox(label="Grounded answer", lines=6)
                reliability = gr.Textbox(label="Reliability signal")

        sources = gr.Dataframe(
            headers=["rank", "title", "doc_id", "retrieval score", "url", "evidence"],
            label="Retrieved evidence",
            wrap=True,
        )
        claims = gr.Dataframe(
            headers=["#", "answer claim", "support label", "term overlap"],
            label="Claim support check",
            wrap=True,
        )
        ask.click(ask_policy_bot, inputs=[question, top_k, threshold], outputs=[answer, reliability, sources, claims])

    with gr.Tab("Evaluation"):
        gr.Markdown(
            """
            Evaluation uses a curated query set with direct, paraphrased, slang/noisy, misleading, negation,
            and unanswerable questions.

            Metrics:
            - Retrieval@k: whether the expected source appears in the retrieved set.
            - Abstention accuracy: whether the system refuses when the corpus cannot answer.
            - Claim support: rough evidence overlap for answer claims.
            """
        )
        eval_button = gr.Button("Run Evaluation", variant="primary")
        eval_summary = gr.Code(label="Evaluation summary", language="json")
        eval_rows = gr.Dataframe(
            headers=[
                "question",
                "stress_type",
                "answerable",
                "expected_doc",
                "retrieved_docs",
                "retrieval",
                "abstention",
                "reliability",
                "claim_support",
            ],
            label="Question-level results",
            wrap=True,
        )
        eval_button.click(run_eval, inputs=[top_k, threshold], outputs=[eval_summary, eval_rows])

    with gr.Tab("Pitch Notes"):
        gr.Markdown(
            """
            **Problem statement:** RAG systems are often trusted because they cite sources, but retrieval does not guarantee
            that generated answers are faithful. In student policy settings, a wrong answer can cause missed deadlines,
            incorrect financial planning, or misunderstanding of conduct obligations.

            **Pretrained model + transfer learning approach:** The retriever uses `sentence-transformers/all-MiniLM-L6-v2`,
            a pretrained sentence embedding model. Instead of fine-tuning weights, the prototype transfers that pretrained
            semantic representation to a Duke policy corpus through vector retrieval. The answer generator optionally uses
            `google/flan-t5-small`; if generation is unavailable, the system falls back to extractive evidence.

            **Data augmentation / stress tests:** The evaluation set includes paraphrases, slang, emojis, negation,
            misleading assumptions, and unanswerable questions.

            **Preliminary results:** Run the Evaluation tab to report Retrieval@k, abstention accuracy, and claim support.
            """
        )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")), share=False)
