from __future__ import annotations

import math
import os
import re
from collections import Counter
from typing import Any


_DENSE_MODEL: Any = None

def retrieve_passages(
    query: str,
    evidence: list[dict[str, Any]],
    method: str,
    limit: int = 12,
) -> list[dict[str, Any]]:
    records = [item for item in evidence if str(item.get("text") or "").strip()]
    if not records:
        return []
    if method == "bm25":
        scores = _bm25(query, [item["text"] for item in records])
    elif method == "tfidf":
        scores = _tfidf(query, [item["text"] for item in records])
    elif method == "dense":
        scores = _dense(query, [item["text"] for item in records])
    else:
        raise ValueError(f"Unknown retrieval method: {method}")
    ranked = sorted(zip(records, scores), key=lambda item: item[1], reverse=True)
    return [
        {
            "evidence_id": item["id"],
            "source": item["source"],
            "type": item["type"],
            "text": item["text"],
            "score": round(float(score), 6),
        }
        for item, score in ranked[:limit]
        if score > 0
    ]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def _bm25(query: str, documents: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    tokenized = [_tokens(document) for document in documents]
    query_terms = _tokens(query)
    average_length = sum(map(len, tokenized)) / max(len(tokenized), 1)
    document_frequency = Counter(
        term for document in tokenized for term in set(document)
    )
    scores = []
    for document in tokenized:
        frequencies = Counter(document)
        score = 0.0
        for term in query_terms:
            frequency = frequencies[term]
            if not frequency:
                continue
            inverse = math.log(
                1 + (len(tokenized) - document_frequency[term] + 0.5)
                / (document_frequency[term] + 0.5)
            )
            denominator = frequency + k1 * (
                1 - b + b * len(document) / max(average_length, 1)
            )
            score += inverse * frequency * (k1 + 1) / denominator
        scores.append(score)
    return scores


def _tfidf(query: str, documents: list[str]) -> list[float]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        return _bm25(query, documents)
    matrix = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit_transform(
        [query, *documents]
    )
    return cosine_similarity(matrix[0:1], matrix[1:]).ravel().tolist()


def _dense(query: str, documents: list[str]) -> list[float]:
    global _DENSE_MODEL
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "Dense retrieval requires the ml dependencies: pip install -e '.[ml]'"
        ) from error
    if _DENSE_MODEL is None:
        allow_download = os.getenv("CLICKLESS_ALLOW_MODEL_DOWNLOAD", "false").lower() in {
            "1",
            "true",
            "yes",
        }
        try:
            _DENSE_MODEL = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                local_files_only=not allow_download,
            )
        except Exception as error:
            raise RuntimeError(
                "MiniLM is not cached. Run python scripts/prepare_models.py once with internet access."
            ) from error
    vectors = _DENSE_MODEL.encode([query, *documents], normalize_embeddings=True)
    return (vectors[1:] @ vectors[0]).tolist()
