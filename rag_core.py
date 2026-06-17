from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "should",
    "that",
    "the",
    "their",
    "to",
    "what",
    "when",
    "with",
    "you",
    "your",
}


@dataclass
class RetrievedChunk:
    doc_id: str
    title: str
    url: str
    category: str
    text: str
    score: float


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def chunk_documents(docs: list[dict[str, Any]], max_words: int = 115, overlap: int = 25) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for doc in docs:
        words = doc["text"].split()
        if len(words) <= max_words:
            chunks.append({**doc, "chunk_id": f"{doc['doc_id']}_0", "chunk_text": doc["text"]})
            continue
        start = 0
        chunk_idx = 0
        while start < len(words):
            end = min(start + max_words, len(words))
            chunks.append(
                {
                    **doc,
                    "chunk_id": f"{doc['doc_id']}_{chunk_idx}",
                    "chunk_text": " ".join(words[start:end]),
                }
            )
            if end == len(words):
                break
            start = max(0, end - overlap)
            chunk_idx += 1
    return chunks


def retrieval_text(chunk: dict[str, Any]) -> str:
    """Text used for retrieval; includes metadata users often mention in queries."""
    return " ".join(
        [
            str(chunk.get("title", "")),
            str(chunk.get("doc_id", "")).replace("_", " "),
            str(chunk.get("category", "")).replace("_", " "),
            str(chunk.get("chunk_text", "")),
        ]
    )


class PolicyRAG:
    def __init__(
        self,
        corpus_path: str | Path = "data/corpus.jsonl",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        generator_model: str = "google/flan-t5-small",
    ):
        self.docs = load_jsonl(corpus_path)
        self.chunks = chunk_documents(self.docs)
        self.embedding_model_name = embedding_model
        self.generator_model_name = generator_model
        self.embedder = None
        self.generator = None
        self.retrieval_texts = [retrieval_text(chunk) for chunk in self.chunks]
        self.tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.tfidf_matrix = self.tfidf.fit_transform(self.retrieval_texts)
        self.embeddings = None
        self._load_embedder()
        self._load_generator()

    def _load_embedder(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self.embedder = SentenceTransformer(self.embedding_model_name)
            self.embeddings = self.embedder.encode(
                self.retrieval_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            print(f"Falling back to TF-IDF retrieval because embedding model failed: {exc}")
            self.embedder = None
            self.embeddings = None

    def _load_generator(self) -> None:
        if os.getenv("DISABLE_GENERATION", "").lower() in {"1", "true", "yes"}:
            return
        try:
            from transformers import pipeline

            self.generator = pipeline("text2text-generation", model=self.generator_model_name, max_new_tokens=140)
        except Exception as exc:
            print(f"Falling back to extractive answers because generator failed: {exc}")
            self.generator = None

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        if self.embedder is not None and self.embeddings is not None:
            query_embedding = self.embedder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
            scores = np.dot(self.embeddings, query_embedding)
        else:
            query_vec = self.tfidf.transform([query])
            scores = cosine_similarity(self.tfidf_matrix, query_vec).ravel()

        scores = self._rerank_scores(query, scores)
        order = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievedChunk(
                doc_id=self.chunks[idx]["doc_id"],
                title=self.chunks[idx]["title"],
                url=self.chunks[idx]["url"],
                category=self.chunks[idx]["category"],
                text=self.chunks[idx]["chunk_text"],
                score=float(scores[idx]),
            )
            for idx in order
        ]

    def _rerank_scores(self, query: str, scores: np.ndarray) -> np.ndarray:
        """Apply lightweight domain reranking for policy-specific wording."""
        adjusted = scores.astype(float).copy()
        query_lower = query.lower()
        query_terms = content_terms(query)
        for idx, chunk in enumerate(self.chunks):
            text = f"{chunk['title']} {chunk['chunk_text']}".lower()
            text_terms = content_terms(text)
            if query_terms:
                adjusted[idx] += 0.08 * (len(query_terms & text_terms) / len(query_terms))

            if "commitment" in query_lower or "commitments" in query_lower:
                if "students commit" in text:
                    adjusted[idx] += 0.30
                if "will not lie, cheat, or steal" in text:
                    adjusted[idx] += 0.20
                if "conduct themselves honorably" in text and "act if the standard is compromised" in text:
                    adjusted[idx] += 0.20

            if "community standard" in query_lower and "to uphold the duke community standard" in text:
                adjusted[idx] += 0.10

            if ("unaware" in query_lower or "did not know" in query_lower or "excuse" in query_lower) and (
                "unawareness of any policy is not a valid excuse" in text
            ):
                adjusted[idx] += 0.35

        return adjusted

    def answer(self, query: str, top_k: int = 3, abstain_threshold: float = 0.28) -> dict[str, Any]:
        retrieved = self.retrieve(query, top_k=top_k)
        top_score = retrieved[0].score if retrieved else 0.0
        if not retrieved or top_score < abstain_threshold:
            answer = "I do not have enough support in the Duke policy corpus to answer that reliably."
            verdict = "abstain"
        else:
            answer = self._generate_or_extract(query, retrieved)
            verdict = "answered"

        support = score_answer_support(answer, " ".join(chunk.text for chunk in retrieved))
        reliability = reliability_label(top_score, support["support_ratio"], verdict)
        return {
            "question": query,
            "answer": answer,
            "verdict": verdict,
            "reliability": reliability,
            "top_score": top_score,
            "support": support,
            "sources": [chunk.__dict__ for chunk in retrieved],
        }

    def _generate_or_extract(self, query: str, retrieved: list[RetrievedChunk]) -> str:
        context = "\n\n".join(f"Source: {chunk.title}\n{chunk.text}" for chunk in retrieved)
        if self.generator is not None:
            prompt = (
                "Answer the question using only the provided Duke policy context. "
                "If the context does not contain the answer, say you do not have enough information.\n\n"
                f"Question: {query}\n\nContext:\n{context}\n\nAnswer:"
            )
            generated = self.generator(prompt)[0]["generated_text"].strip()
            if generated:
                return generated

        sentences = split_sentences(retrieved[0].text)
        query_terms = content_terms(query)
        best = max(sentences, key=lambda sentence: len(content_terms(sentence) & query_terms), default=retrieved[0].text)
        return best


def split_sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]


def content_terms(text: str) -> set[str]:
    terms = re.findall(r"[a-zA-Z][a-zA-Z0-9-]+", text.lower())
    return {term for term in terms if term not in STOPWORDS and len(term) > 2}


def score_answer_support(answer: str, evidence: str) -> dict[str, Any]:
    claims = split_sentences(answer)
    evidence_terms = content_terms(evidence)
    claim_results = []
    supported_count = 0
    for claim in claims:
        terms = content_terms(claim)
        if not terms:
            overlap = 0.0
        else:
            overlap = len(terms & evidence_terms) / len(terms)
        supported = overlap >= 0.55
        supported_count += int(supported)
        claim_results.append({"claim": claim, "term_overlap": round(overlap, 3), "supported": supported})
    support_ratio = supported_count / max(len(claims), 1)
    return {"support_ratio": support_ratio, "claims": claim_results}


def reliability_label(top_score: float, support_ratio: float, verdict: str) -> str:
    if verdict == "abstain":
        return "Not enough evidence"
    if top_score >= 0.5 and support_ratio >= 0.8:
        return "High"
    if top_score >= 0.32 and support_ratio >= 0.5:
        return "Medium"
    return "Low"
