from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from typing import Any, Iterable


AUTOMOTIVE_LEXICON = {
    "reliable": 2.0, "dependable": 2.0, "durable": 1.8, "roomy": 1.5, "spacious": 1.5,
    "efficient": 1.5, "economical": 1.5, "breakdown": -2.5, "broke": -2.0, "recall": -1.5,
    "leak": -2.0, "leaks": -2.0, "rust": -2.0, "rusted": -2.0, "squeak": -1.5, "squeaks": -1.5,
    "rattle": -1.5, "rattles": -1.5, "stalled": -2.5, "stalls": -2.5, "overheated": -2.5,
    "cramped": -1.5, "guzzler": -2.0, "pricey": -1.2, "costly": -1.8,
}

POLARITY_SCORE = {
    "positive": 1.0,
    "negative": 0.0,
    "neutral": 0.5,
}


def vader_labels(texts: Iterable[str]) -> list[str]:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError as error:
        raise RuntimeError(
            "VADER requires the ml dependencies: pip install -e '.[ml]'"
        ) from error
    analyzer = SentimentIntensityAnalyzer()
    analyzer.lexicon.update(AUTOMOTIVE_LEXICON)
    labels = []
    for text in texts:
        compound = analyzer.polarity_scores(text)["compound"]
        labels.append("positive" if compound >= 0.05 else "negative" if compound <= -0.05 else "neutral")
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
        grouped.setdefault(key, {}).setdefault(assertion["aspect"], []).append(POLARITY_SCORE[label])
    for key, values_by_aspect in grouped.items():
        review = reviews[key]
        review["aspects"] = {
            aspect: round(sum(values) / len(values), 4)
            for aspect, values in values_by_aspect.items()
        }
        review["aspect_method"] = "vader"
    return updated
