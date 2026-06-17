---
title: Duke Policy RAG Reliability Checker
emoji: 🧭
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---

# Duke Policy RAG Reliability Checker

Mini Hackathon #2 project: **Can Machines Understand Us Reliably?**

This prototype tests whether a small RAG assistant can answer Duke student policy
questions with grounded evidence instead of unsupported confidence. The app is
designed for Hugging Face Spaces and includes both an interactive question-answer
flow and an evaluation dashboard.

## Problem Statement

Students often ask policy questions in informal language: financial aid deadlines,
summer aid, study away, and conduct expectations. A wrong or hallucinated answer
can cause missed deadlines, incorrect planning, or misunderstanding of university
responsibilities. This project explores whether a RAG system can answer these
questions reliably and show when it should abstain.

## NLP Approach

- **Retrieval model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Transfer learning approach:** use pretrained sentence embeddings as a feature
  extractor over a curated Duke policy corpus.
- **Generation model:** optional `google/flan-t5-small`; the app falls back to an
  extractive answer when generation is unavailable.
- **Reliability layer:** simple claim-support check against retrieved evidence.

This is intentionally small and inspectable for a hackathon. It is not an
official Duke tool.

## Data

The corpus is curated from public Duke policy pages:

- Karsh Office of Undergraduate Financial Support: Apply for Aid
- Karsh Office of Undergraduate Financial Support: Renew Your Aid
- Karsh Office of Undergraduate Financial Support: Summer Study
- Karsh Office of Undergraduate Financial Support: Study Away
- Duke Community Standard homepage
- Duke Community Standard policies page

The corpus lives in:

```text
data/corpus.jsonl
```

Each record includes:

```json
{
  "doc_id": "aid_renewal",
  "title": "Renew Your Aid",
  "url": "https://financialaid.duke.edu/apply-aid/renew-your-aid/",
  "category": "financial_aid",
  "text": "..."
}
```

## Evaluation Plan

The evaluation set is in:

```text
data/eval_questions.jsonl
```

It includes direct, paraphrased, slang/noisy, misleading, negation, and
unanswerable questions.

Metrics:

- **Retrieval@k:** whether the expected source appears in the top-k retrieved
  chunks.
- **Abstention accuracy:** whether the system refuses when the corpus does not
  contain enough evidence.
- **Mean claim support:** rough overlap-based check of whether answer claims are
  grounded in retrieved evidence.
- **Stress-test breakdown:** performance by question type.

Run:

```bash
python evaluate.py
```

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Then open the local Gradio URL.

To disable generative answers and use extractive-only mode:

```bash
DISABLE_GENERATION=true python app.py
```

## Deploy To Hugging Face Spaces

1. Create a new Hugging Face Space.
2. Choose **Gradio**.
3. Push this repo to the Space.
4. Hugging Face will install `requirements.txt` and run `app.py`.

CLI sketch:

```bash
git init
git add .
git commit -m "Initial Duke policy RAG reliability prototype"
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/duke-policy-rag-reliability
git push space main
```

## Pitch Checklist

1. **Problem statement:** Student policy RAG answers can be harmful if unsupported
   or overconfident.
2. **Pretrained model + transfer learning:** pretrained MiniLM sentence
   embeddings transferred to Duke policy retrieval; optional FLAN-T5 generation.
3. **Data augmentation / stress tests:** paraphrases, slang, emojis, negation,
   misleading assumptions, and unanswerable questions.
4. **Preliminary results:** report Retrieval@k, abstention accuracy, claim
   support, and examples where evaluation reveals limitations.

## Limitations

- The corpus is intentionally small and curated.
- The claim-support check is a simple lexical heuristic, not a formal factuality
  model.
- Generated answers should be verified against official Duke sources.
- This project is for educational/hackathon use only.
