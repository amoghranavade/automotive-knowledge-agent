from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

from clickless.models import Constraints
from clickless.verifier import verify_candidate
from .qwen import QwenResponseError


ASPECTS = {"reliability", "fuel_economy", "cargo", "maintenance"}
FEATURES = ASPECTS | {"price", "mileage", "safety", "snow"}
POLARITIES = {"positive", "negative", "neutral", "mixed"}
SUBSYSTEMS = {
    "engine",
    "transmission",
    "brakes",
    "suspension",
    "electrical",
    "battery",
    "cabin",
    "drivetrain",
    "body",
    "steering",
    "fuel_system",
    "airbags",
    "tires",
    "unknown",
}
BODY_STYLES = {"suv", "sedan", "truck"}
DRIVETRAINS = {"AWD", "4WD", "FWD", "RWD"}
POLARITY_SCORE = {
    "positive": 1.0,
    "negative": 0.0,
    "neutral": 0.5,
    "mixed": 0.5,
}


def parse_request_with_model(
    client: Any,
    query: str,
    fallback: Constraints,
    fallback_questions: list[str],
) -> tuple[Constraints, list[str]]:
    system = (
        "Convert a used-car request to JSON. Keep unstated hard limits null. "
        "Allowed body_style values: suv, sedan, truck. Allowed drivetrain values: "
        "AWD, 4WD, FWD, RWD. Preference keys: price, mileage, safety, reliability, "
        "maintenance, fuel_economy, cargo, snow. Return numeric preference weights "
        "between 0 and 1 and at most two short clarification questions."
    )
    user = json.dumps(
        {
            "request": query,
            "output": {
                "max_price": "integer or null",
                "max_mileage": "integer or null",
                "min_year": "integer or null",
                "body_style": "string or null",
                "drivetrain": "string or null",
                "preferences": {"feature": "number"},
                "clarification_questions": ["string"],
            },
        }
    )
    payload = client.complete_json(system, user, max_tokens=700)
    constraints = Constraints(
        max_price=_positive_int(payload.get("max_price"), fallback.max_price),
        max_mileage=_positive_int(payload.get("max_mileage"), fallback.max_mileage),
        min_year=_year(payload.get("min_year"), fallback.min_year),
        body_style=_allowed(payload.get("body_style"), BODY_STYLES, fallback.body_style),
        drivetrain=_allowed(payload.get("drivetrain"), DRIVETRAINS, fallback.drivetrain),
        preferences=_preferences(payload.get("preferences"), fallback.preferences),
    )
    raw_questions = payload.get("clarification_questions")
    model_questions = (
        [str(item).strip() for item in raw_questions if str(item).strip()][:2]
        if isinstance(raw_questions, list)
        else fallback_questions
    )
    required_questions = []
    if constraints.max_price is None:
        required_questions.append("What is your maximum budget?")
    if constraints.body_style is None:
        required_questions.append("Which body style do you prefer: SUV, sedan, or truck?")
    questions = list(dict.fromkeys(required_questions + model_questions))[:2]
    return constraints, questions


def apply_model_aspects(
    client: Any,
    vehicles: list[dict[str, Any]],
    max_reviews_per_vehicle: int = 1,
) -> list[dict[str, Any]]:
    reviews = [
        {"source_id": review["source_id"], "text": review["text"]}
        for vehicle in vehicles
        for review in [
            item
            for item in vehicle.get("reviews", [])
            if item.get("match_type") == "exact_year"
        ][:max_reviews_per_vehicle]
    ]
    if not reviews:
        return vehicles

    system = (
        "Extract review evidence as JSON. Use only exact text spans from each review. "
        "Allowed aspects: reliability, fuel_economy, cargo, maintenance. Allowed "
        "polarities: positive, negative, neutral, mixed. Allowed subsystems: engine, "
        "transmission, brakes, suspension, electrical, battery, cabin, drivetrain, "
        "body, steering, fuel_system, airbags, tires, unknown. Do not add claims not "
        "present in the text. Return at most three assertions per review."
    )
    review_text = {item["source_id"]: item["text"] for item in reviews}
    extracted: dict[str, dict[str, list[float]]] = {}
    assertions_by_review: dict[str, list[dict[str, Any]]] = {}
    for start in range(0, len(reviews), 24):
        user = json.dumps(
            {
                "reviews": reviews[start : start + 24],
                "output": {
                    "reviews": [
                        {
                            "source_id": "string",
                            "assertions": [
                                {
                                    "aspect": "string",
                                    "subsystem": "string",
                                    "polarity": "string",
                                    "span": "exact review substring",
                                }
                            ],
                        }
                    ]
                },
            }
        )
        payload = client.complete_json(system, user, max_tokens=1200)
        response_reviews = payload.get("reviews")
        if not isinstance(response_reviews, list):
            raise QwenResponseError("Qwen review response is missing a reviews list")
        for item in response_reviews:
            if not isinstance(item, dict):
                continue
            source_id = str(item.get("source_id", ""))
            text = review_text.get(source_id)
            if text is None:
                continue
            assertions = item.get("assertions")
            if not isinstance(assertions, list):
                continue
            for assertion in assertions:
                if not isinstance(assertion, dict):
                    continue
                aspect = str(assertion.get("aspect", ""))
                polarity = str(assertion.get("polarity", ""))
                subsystem = str(assertion.get("subsystem", ""))
                span = str(assertion.get("span", "")).strip()
                if (
                    aspect not in ASPECTS
                    or polarity not in POLARITIES
                    or subsystem not in SUBSYSTEMS
                    or not span
                    or span.casefold() not in text.casefold()
                ):
                    continue
                extracted.setdefault(source_id, {}).setdefault(aspect, []).append(
                    POLARITY_SCORE[polarity]
                )
                assertions_by_review.setdefault(source_id, []).append(
                    {
                        "aspect": aspect,
                        "subsystem": subsystem,
                        "polarity": polarity,
                        "span": span,
                        "method": "qwen",
                    }
                )

    updated = deepcopy(vehicles)
    for vehicle in updated:
        for review in vehicle.get("reviews", []):
            aspects = extracted.get(review["source_id"])
            if aspects:
                review["aspects"] = {
                    name: round(sum(values) / len(values), 4)
                    for name, values in aspects.items()
                }
                review["aspect_assertions"] = assertions_by_review.get(review["source_id"], [])
                review["aspect_method"] = "qwen"
    return updated


def apply_grounded_summaries(
    client: Any,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not candidates:
        return candidates
    records = []
    for candidate in candidates:
        context_evidence = _summary_evidence(candidate)
        context_ids = {item["id"] for item in context_evidence}
        records.append(
            {
                "candidate_id": candidate["id"],
                "vehicle": {
                    "year": candidate["year"],
                    "make": candidate["make"],
                    "model": candidate["model"],
                    "trim": candidate["trim"],
                },
                "evidence": [
                    {
                        "id": item["id"],
                        "text": item["text"],
                        "indirect": item["indirect"],
                    }
                    for item in context_evidence
                ],
                "graph_paths": [
                    {
                        "kind": path["kind"],
                        "label": path["label"],
                        "component": path.get("component"),
                        "text": path["text"],
                        "evidence_ids": [
                            evidence_id
                            for evidence_id in path["evidence_ids"]
                            if evidence_id in context_ids
                        ],
                        "indirect": path["indirect"],
                    }
                    for path in candidate.get("graph_paths", [])
                    if any(
                        evidence_id in context_ids
                        for evidence_id in path.get("evidence_ids", [])
                    )
                ],
            }
        )
    system = (
        "Write short used-car evidence summaries as JSON using the supplied graph paths "
        "to connect related records. Every statement must include "
        "one or more supports containing a supplied evidence ID and an exact quote from "
        "that evidence. Do not create or modify prices, mileage, ratings, "
        "recall counts, fitment, or repair costs. Do not present indirect evidence as an "
        "exact-year fact. Return one strength and one caution per vehicle."
    )
    user = json.dumps(
        {
            "candidates": records,
            "output": {
                "candidates": [
                    {
                        "candidate_id": "string",
                        "strengths": [
                            {
                                "text": "string",
                                "supports": [
                                    {"evidence_id": "string", "quote": "exact substring"}
                                ],
                            }
                        ],
                        "cautions": [
                            {
                                "text": "string",
                                "supports": [
                                    {"evidence_id": "string", "quote": "exact substring"}
                                ],
                            }
                        ],
                    }
                ]
            },
        }
    )
    payload = client.complete_json(system, user, max_tokens=1200)
    by_id = {candidate["id"]: candidate for candidate in deepcopy(candidates)}
    response_candidates = payload.get("candidates")
    if not isinstance(response_candidates, list):
        raise QwenResponseError("Qwen summary response is missing a candidates list")
    for item in response_candidates:
        if not isinstance(item, dict):
            continue
        candidate = by_id.get(str(item.get("candidate_id", "")))
        if candidate is None:
            continue
        evidence_by_id = {
            evidence["id"]: evidence for evidence in candidate["evidence"]
        }
        strength_ids = {
            evidence["id"]
            for evidence in candidate["evidence"]
            if evidence["type"] in {"listing", "safety", "review"}
            and not evidence["indirect"]
        }
        strengths = _claims(item.get("strengths"), evidence_by_id, strength_ids)
        cautions = _claims(item.get("cautions"), evidence_by_id, set(evidence_by_id))
        if strengths:
            candidate["pros"] = strengths
        if cautions:
            candidate["cautions"] = _dedupe_claims(candidate["cautions"] + cautions)
        verify_candidate(candidate)
    return [by_id[candidate["id"]] for candidate in candidates]


def _summary_evidence(
    candidate: dict[str, Any],
    max_records: int = 4,
    max_chars: int = 2_500,
) -> list[dict[str, Any]]:
    by_id = {item["id"]: item for item in candidate["evidence"]}
    ordered_ids = []
    ordered_ids.extend(
        item["evidence_id"] for item in candidate.get("retrieved_passages", [])
    )
    for path in candidate.get("graph_paths", []):
        ordered_ids.extend(path.get("evidence_ids", []))
    ordered_ids.extend(
        item["id"]
        for item in candidate["evidence"]
        if item["type"] in {"listing", "safety"}
    )
    ordered_ids.extend(item["id"] for item in candidate["evidence"])
    selected = []
    used = 0
    for evidence_id in dict.fromkeys(ordered_ids):
        item = by_id.get(evidence_id)
        if item is None or len(selected) == max_records or used >= max_chars:
            continue
        text = str(item.get("text") or "")
        remaining = max_chars - used
        if remaining <= 0:
            break
        record = dict(item)
        record["text"] = text[:remaining]
        selected.append(record)
        used += len(record["text"])
    return selected


def _positive_int(value: Any, fallback: int | None) -> int | None:
    if isinstance(value, bool):
        return fallback
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed > 0 else fallback


def _year(value: Any, fallback: int | None) -> int | None:
    parsed = _positive_int(value, fallback)
    return parsed if parsed is None or 1980 <= parsed <= 2100 else fallback


def _allowed(value: Any, allowed: set[str], fallback: str | None) -> str | None:
    if not isinstance(value, str):
        return fallback
    for candidate in (value, value.lower(), value.upper()):
        if candidate in allowed:
            return candidate
    return fallback


def _preferences(value: Any, fallback: dict[str, float]) -> dict[str, float]:
    if not isinstance(value, dict):
        return fallback
    weights = {}
    for name, weight in value.items():
        if name not in FEATURES or isinstance(weight, bool):
            continue
        try:
            parsed = float(weight)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            weights[name] = min(parsed, 1.0)
    total = sum(weights.values())
    if not total:
        return fallback
    return {name: round(weight / total, 4) for name, weight in weights.items()}


def _claims(
    value: Any,
    evidence_by_id: dict[str, dict[str, Any]],
    allowed_ids: set[str],
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    claims = []
    for item in value[:2]:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        supports = item.get("supports")
        if not text or not isinstance(supports, list):
            continue
        evidence_ids = []
        for support in supports:
            if not isinstance(support, dict):
                continue
            evidence_id = str(support.get("evidence_id", ""))
            quote = str(support.get("quote", "")).strip()
            evidence = evidence_by_id.get(evidence_id)
            if (
                evidence_id not in allowed_ids
                or evidence is None
                or not quote
                or quote.casefold() not in evidence["text"].casefold()
            ):
                continue
            if evidence_id not in evidence_ids:
                evidence_ids.append(evidence_id)
        if evidence_ids:
            claims.append({"text": text, "evidence_ids": evidence_ids})
    return claims


def _dedupe_claims(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique = []
    seen = set()
    for claim in claims:
        key = claim["text"].casefold()
        if key not in seen:
            unique.append(claim)
            seen.add(key)
    return unique
