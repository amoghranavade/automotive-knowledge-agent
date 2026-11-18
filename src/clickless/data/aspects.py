from __future__ import annotations

import re
from typing import Any


ASPECT_TERMS = {
    "reliability": {"reliable", "reliability", "dependable", "breakdown", "failed", "failure"},
    "fuel_economy": {"mpg", "fuel", "gas mileage", "economy", "efficient"},
    "cargo": {
        "cargo", "trunk", "storage", "space", "room", "roomy", "roominess",
        "legroom", "headroom", "spacious",
    },
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
    "spacious", "roomy", "plenty", "inexpensive", "sturdy", "solid",
}
NEGATIVE = {
    "bad", "poor", "failed", "failure", "broken", "expensive", "problem", "issue",
    "worst", "hate", "rough", "unreliable", "costly", "leak", "noise",
    "cramped", "disappointing", "terrible",
}


NEGATORS = {
    "not", "no", "never", "isn't", "isnt", "wasn't", "wasnt", "aren't", "arent",
    "don't", "dont", "doesn't", "doesnt", "didn't", "didnt", "hardly", "barely",
    "nothing", "without",
}
# Phrases that say the owner can't judge yet, so the sentence carries no verdict.
UNCERTAIN = ("too early to tell", "time will tell", "remains to be seen", "can't say yet")
# Phrases whose sentiment word means the opposite of its usual sense.
PHRASE_OVERRIDES = (
    (re.compile(r"\bexpensive[-\s]looking\b"), "premium-looking"),
    (re.compile(r"\bno (?:problems?|issues?)\b"), "trouble-free"),
)
_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _polarity(lowered: str) -> str:
    """Count sentiment words, flipping any that follow a negator within 3 words."""
    if any(phrase in lowered for phrase in UNCERTAIN):
        return "neutral"
    for pattern, replacement in PHRASE_OVERRIDES:
        lowered = pattern.sub(replacement, lowered)
    tokens = _TOKEN_RE.findall(lowered)
    positive = negative = 0
    for index, token in enumerate(tokens):
        is_positive = token in POSITIVE or token == "trouble"
        is_negative = token in NEGATIVE
        if token == "trouble":
            # "trouble-free" was split into "trouble", "free"
            is_positive = index + 1 < len(tokens) and tokens[index + 1] == "free"
            is_negative = not is_positive
        if not (is_positive or is_negative):
            continue
        negated = any(word in NEGATORS for word in tokens[max(0, index - 3):index])
        if is_positive != negated:
            positive += 1
        else:
            negative += 1
    if positive and negative:
        return "mixed"
    if positive:
        return "positive"
    if negative:
        return "negative"
    return "neutral"


def _mentions(lowered: str, term: str) -> bool:
    # Whole-word match so "room" does not fire inside "bathroom".
    return (
        re.search(rf"(?<![a-z]){re.escape(term)}(?:s|es|d|ed|ing)?(?![a-z])", lowered)
        is not None
    )


def extract_aspects(text: str, prefix: str) -> list[dict[str, Any]]:
    assertions = []
    sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+|\n+", text) if item.strip()]
    for sentence_index, sentence in enumerate(sentences):
        lowered = sentence.casefold()
        # "fuel pump" / "fuel tank" are parts, not fuel economy.
        economy_text = re.sub(r"\bfuel (?:pump|system|tank|filter|injectors?|line)\b", "", lowered)
        aspects = [
            name for name, terms in ASPECT_TERMS.items()
            if any(
                _mentions(economy_text if name == "fuel_economy" else lowered, term)
                for term in terms
            )
        ]
        if not aspects:
            continue
        polarity = _polarity(lowered)
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
