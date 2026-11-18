from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import re
from typing import Any, Iterable


POLARITY_SCORE = {
    "positive": 1.0,
    "negative": 0.0,
    "neutral": 0.5,
    "mixed": 0.5,
}

AUTOMOTIVE_LEXICON = {
    "reliable": 2.0,
    "dependable": 2.2,
    "durable": 1.8,
    "bulletproof": 2.5,
    "fuel-efficient": 2.0,
    "efficient": 1.5,
    "economical": 1.5,
    "affordable": 1.8,
    "roomy": 1.5,
    "spacious": 1.8,
    "smooth": 1.6,
    "wore": -1.8,
    "costly": -2.0,
    "pricey": -1.2,
    "overpriced": -2.0,
    "rust": -2.0,
    "rusted": -2.2,
    "leaking": -2.2,
    "leak": -1.8,
    "leaks": -2.0,
    "squeak": -1.5,
    "squeaks": -1.5,
    "rattle": -1.5,
    "rattles": -1.5,
    "stalled": -2.2,
    "stalls": -2.5,
    "overheated": -2.5,
    "cramped": -1.5,
    "guzzler": -2.0,
    "jerky": -1.8,
    "clunk": -1.6,
    "recall": -1.2,
    "unreliable": -2.5,
    "broke": -2.0,
    "breakdown": -2.4,
    "high_mileage": -1.5,
}

_CONTRASTIVE_RE = re.compile(r"\b(?:but|however|although|though|yet|while)\b", re.I)
_LEADING_CONTRASTIVE_RE = re.compile(r"^\s*(?:although|though|while)\b", re.I)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _normalize_automotive_text(text: str) -> str:
    normalized = re.sub(
        r"\b(good|great|excellent|solid|nice)\s+(?:(?:gas|fuel)\s+)?(?:mileage|mpg)\b",
        r"\1 fuel-efficient",
        text,
        flags=re.I,
    )
    return re.sub(r"\bhigh[-\s]+mileage\b", "high_mileage", normalized, flags=re.I)


def _contrastive_clauses(sentence: str) -> list[str]:
    stripped = sentence.strip()
    if not stripped:
        return []
    if _LEADING_CONTRASTIVE_RE.match(stripped):
        body = _LEADING_CONTRASTIVE_RE.sub("", stripped, count=1).strip(" ,;:")
        if "," in body:
            first, rest = body.split(",", 1)
            parts = [first, *_CONTRASTIVE_RE.split(rest)]
            return [part.strip(" ,;:") for part in parts if part.strip(" ,;:")]
    parts = _CONTRASTIVE_RE.split(stripped)
    return [part.strip(" ,;:") for part in parts if part.strip(" ,;:")]


def _vader_analyzer() -> Any:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError as error:
        raise RuntimeError(
            "VADER requires the ml dependencies: pip install -e '.[ml]'"
        ) from error
    analyzer = SentimentIntensityAnalyzer()
    analyzer.lexicon.update(AUTOMOTIVE_LEXICON)
    return analyzer


def vader_labels(texts: Iterable[str]) -> list[str]:
    analyzer = _vader_analyzer()
    labels = []
    for text in texts:
        normalized = _normalize_automotive_text(str(text or ""))
        is_mixed = False
        sentences = [item.strip() for item in _SENTENCE_SPLIT_RE.split(normalized) if item.strip()]
        clause_groups = [_contrastive_clauses(sentence) for sentence in sentences]
        if len(sentences) > 1 and _CONTRASTIVE_RE.search(normalized):
            clause_groups.append(
                [part.strip(" ,;:") for part in _CONTRASTIVE_RE.split(normalized) if part.strip(" ,;:")]
            )
        for clauses in clause_groups:
            if len(clauses) >= 2:
                compounds = [analyzer.polarity_scores(clause)["compound"] for clause in clauses]
                if any(score >= 0.05 for score in compounds) and any(
                    score <= -0.05 for score in compounds
                ):
                    is_mixed = True
                    break
        if is_mixed:
            labels.append("mixed")
            continue
        compound = analyzer.polarity_scores(normalized)["compound"]
        labels.append(
            "positive"
            if compound >= 0.05
            else "negative"
            if compound <= -0.05
            else "neutral"
        )
    return labels


@dataclass
class TfidfLogisticSentiment:
    vectorizer: Any | None = None
    classifier: Any | None = None

    def fit(self, texts: list[str], labels: list[str]) -> "TfidfLogisticSentiment":
        if len(texts) != len(labels) or not texts:
            raise ValueError("Training texts and labels must be non-empty and aligned")
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=25_000)
        matrix = self.vectorizer.fit_transform(texts)
        self.classifier = LogisticRegression(
            max_iter=1_000,
            class_weight="balanced",
            random_state=42,
        ).fit(matrix, labels)
        return self

    def predict(self, texts: list[str]) -> list[str]:
        if self.vectorizer is None or self.classifier is None:
            raise RuntimeError("The sentiment model has not been fitted")
        matrix = self.vectorizer.transform(texts)
        return self.classifier.predict(matrix).tolist()


def apply_vader_aspects(vehicles: list[dict]) -> list[dict]:
    updated = deepcopy(vehicles)
    spans = []
    targets = []
    for vehicle in updated:
        for review in vehicle.get("reviews", []):
            for assertion in review.get("aspect_assertions", []):
                span = str(assertion.get("span") or "").strip()
                if span:
                    spans.append(span)
                    targets.append((review, assertion))
    if not spans:
        return updated
    grouped: dict[int, dict[str, list[float]]] = {}
    reviews: dict[int, dict] = {}
    for (review, assertion), label in zip(targets, vader_labels(spans)):
        assertion["polarity"] = label
        assertion["method"] = "vader"
        key = id(review)
        reviews[key] = review
        grouped.setdefault(key, {}).setdefault(assertion["aspect"], []).append(
            POLARITY_SCORE[label]
        )
    for key, values_by_aspect in grouped.items():
        review = reviews[key]
        review["aspects"] = {
            aspect: round(sum(values) / len(values), 4)
            for aspect, values in values_by_aspect.items()
        }
        review["aspect_method"] = "vader"
    return updated
