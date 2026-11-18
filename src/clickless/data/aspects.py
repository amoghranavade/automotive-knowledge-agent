from __future__ import annotations

import re
from typing import Any


ASPECT_TERMS = {
    "reliability": {"reliable", "reliability", "dependable", "breakdown", "failed", "failure"},
    "fuel_economy": {"mpg", "fuel", "gas mileage", "economy", "efficient"},
    "cargo": {"cargo", "trunk", "storage", "space", "room"},
    "maintenance": {"maintenance", "repair", "service", "cost", "replace", "replacement"},
}
SUBSYSTEM_TERMS = {
    "engine": {"engine", "motor", "oil"},
    "transmission": {"transmission", "gear", "shifting", "cvt"},
    "brakes": {"brake", "brakes", "rotor", "pad"},
    "suspension": {"suspension", "strut", "shock"},
    "electrical": {"electrical", "alternator", "starter", "sensor"},
    "battery": {"battery", "charging"},
    "cabin": {"infotainment", "stereo", "screen", "camera", "cabin"},
    "drivetrain": {"awd", "4wd", "drivetrain", "differential"},
    "body": {"body", "paint", "rust", "door"},
    "steering": {"steering", "steer", "power steering"},
    "fuel_system": {"fuel pump", "fuel system", "gas tank", "fuel tank"},
    "airbags": {"air bag", "airbag", "inflator"},
    "tires": {"tire", "tyre", "wheel"},
}
POSITIVE = {
    "good", "great", "excellent", "reliable", "dependable", "easy", "efficient",
    "comfortable", "love", "best", "smooth", "affordable", "durable",
}
NEGATIVE = {
    "bad", "poor", "failed", "failure", "broken", "expensive", "problem", "issue",
    "worst", "hate", "rough", "unreliable", "costly", "leak", "noise",
}


def extract_aspects(text: str, prefix: str) -> list[dict[str, Any]]:
    assertions = []
    sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+|\n+", text) if item.strip()]
    for sentence_index, sentence in enumerate(sentences):
        lowered = sentence.casefold()
        words = set(re.findall(r"[a-z0-9]+", lowered))
        aspects = [
            name for name, terms in ASPECT_TERMS.items()
            if any(term in lowered for term in terms)
        ]
        if not aspects:
            continue
        positive = len(words & POSITIVE)
        negative = len(words & NEGATIVE)
        polarity = "mixed" if positive and negative else "positive" if positive else "negative" if negative else "neutral"
        subsystem = next(
            (
                name for name, terms in SUBSYSTEM_TERMS.items()
                if any(term in lowered for term in terms)
            ),
            "unknown",
        )
        for aspect in aspects:
            assertions.append(
                {
                    "id": f"{prefix}-aspect-{sentence_index}-{aspect}",
                    "aspect": aspect,
                    "subsystem": subsystem,
                    "polarity": polarity,
                    "span": sentence[:700],
                    "method": "lexicon",
                }
            )
    return assertions


def normalize_subsystem(text: str) -> str:
    lowered = str(text or "").casefold()
    for name, terms in SUBSYSTEM_TERMS.items():
        if any(term in lowered for term in terms):
            return name
    return "unknown"
