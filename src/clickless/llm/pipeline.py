from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import threading
from typing import Any

from clickless.models import Constraints
from clickless.parser import normalize_make
from clickless.verifier import VerificationError, _verify_numeric_claims, verify_candidate
from .qwen import QwenError, QwenResponseError


ASPECTS = {"reliability", "fuel_economy", "cargo", "maintenance"}
BASELINE_FEATURES = {"price", "mileage", "safety", "reliability"}
OPTIONAL_FEATURES = {"maintenance", "fuel_economy", "cargo", "snow"}
FEATURES = BASELINE_FEATURES | OPTIONAL_FEATURES
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

_CACHE_LOCK = threading.Lock()
_PARSE_CACHE: dict[tuple[str, str], tuple[Constraints, list[str]]] = {}
_ASPECT_CACHE: dict[tuple[str, str, str], list[dict[str, str]]] = {}


def _model_key(client: Any) -> str:
    return str(getattr(client, "model", "") or "")


def parse_request_with_model(
    client: Any,
    query: str,
    fallback: Constraints,
    fallback_questions: list[str],
) -> tuple[Constraints, list[str]]:
    cache_key = (_model_key(client), query)
    with _CACHE_LOCK:
        if cache_key in _PARSE_CACHE:
            cached_constraints, cached_questions = _PARSE_CACHE[cache_key]
            return cached_constraints, list(cached_questions)

    system = (
        "Convert a used-car request to JSON. Keep unstated hard limits (max_price, "
        "max_mileage, min_year, body_style, drivetrain, make) null — do not infer "
        "drivetrain from snow or make. Allowed body_style values: suv, sedan, truck. "
        "Allowed drivetrain values: AWD, 4WD, FWD, RWD. Normalize make to standard "
        "brand names (e.g., Toyota, Chevrolet, Volkswagen). Preference keys: price, "
        "mileage, safety, reliability, maintenance, fuel_economy, cargo, snow. "
        "Assign positive weights (0.15-0.4) to the four baseline universal features "
        "(price, mileage, safety, reliability), boosting any emphasized feature; assign "
        "0 to unmentioned optional features (maintenance, fuel_economy, cargo, snow). "
        "Include at most two short clarification questions."
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
                "make": "string or null",
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
        make=normalize_make(payload.get("make")) or fallback.make,
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
    with _CACHE_LOCK:
        _PARSE_CACHE[cache_key] = (constraints, list(questions))
    return constraints, questions


def _select_reviews_for_extraction(
    vehicle: dict[str, Any],
    max_reviews_per_vehicle: int,
) -> list[dict[str, Any]]:
    if max_reviews_per_vehicle <= 0:
        return []
    owner_reviews = [
        item
        for item in vehicle.get("reviews", [])
        if item.get("match_type") == "exact_year"
        and item.get("source_kind", "owner_review") != "product_review"
    ]
    product_reviews = [
        item
        for item in vehicle.get("reviews", [])
        if item.get("source_kind") == "product_review"
    ]
    product_quota = min(len(product_reviews), max(0, max_reviews_per_vehicle // 3))
    owner_take = owner_reviews[: max_reviews_per_vehicle - product_quota]
    product_take = product_reviews[: max_reviews_per_vehicle - len(owner_take)]
    if len(owner_take) + len(product_take) < max_reviews_per_vehicle:
        extra = max_reviews_per_vehicle - len(owner_take) - len(product_take)
        owner_take = owner_reviews[: len(owner_take) + extra]
    return owner_take + product_take


def _laplace_review_aspects(assertions: list[dict[str, str]]) -> dict[str, float]:
    counts: dict[str, dict[str, int]] = {}
    for assertion in assertions:
        bucket = counts.setdefault(
            assertion["aspect"],
            {"positive": 0, "negative": 0, "total": 0},
        )
        polarity = assertion["polarity"]
        if polarity == "positive":
            bucket["positive"] += 1
        elif polarity == "negative":
            bucket["negative"] += 1
        bucket["total"] += 1
    return {
        aspect: round(
            (((values["positive"] - values["negative"]) / (values["total"] + 1)) + 1.0)
            / 2.0,
            4,
        )
        for aspect, values in counts.items()
    }


def apply_model_aspects(
    client: Any,
    vehicles: list[dict[str, Any]],
    max_reviews_per_vehicle: int = 2,
) -> list[dict[str, Any]]:
    model_name = _model_key(client)
    pending_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    target_keys_by_vehicle: list[set[tuple[str, str, str]]] = []

    with _CACHE_LOCK:
        cached_keys = set(_ASPECT_CACHE)
    for vehicle in vehicles:
        selected = _select_reviews_for_extraction(vehicle, max_reviews_per_vehicle)
        keys: set[tuple[str, str, str]] = set()
        for review in selected:
            source_id = str(review["source_id"])
            text = str(review["text"])
            cache_key = (model_name, source_id, text)
            keys.add(cache_key)
            if cache_key not in cached_keys and cache_key not in pending_by_key:
                pending_by_key[cache_key] = {"source_id": source_id, "text": text[:500]}
        target_keys_by_vehicle.append(keys)

    if not any(target_keys_by_vehicle):
        return vehicles

    pending = list(pending_by_key.items())
    if pending:
        system = (
            "Extract review evidence as JSON. Use only exact text spans from each review. "
            "Allowed aspects: reliability, fuel_economy, cargo, maintenance. Allowed "
            "polarities: positive, negative, neutral, mixed. Allowed subsystems: engine, "
            "transmission, brakes, suspension, electrical, battery, cabin, drivetrain, "
            "body, steering, fuel_system, airbags, tires, unknown. Do not add claims not "
            "present in the text."
        )
        batches = [pending[start : start + 3] for start in range(0, len(pending), 3)]

        def _extract_batch(
            batch_items: list[tuple[tuple[str, str, str], dict[str, str]]],
        ) -> list[tuple[tuple[str, str, str], list[dict[str, str]]]]:
            batch_payload = [item for _, item in batch_items]
            key_by_source_id = {key[1]: key for key, _ in batch_items}
            text_by_source_id = {key[1]: key[2] for key, _ in batch_items}
            extracted_batch: dict[str, list[dict[str, str]]] = {
                sid: [] for sid in key_by_source_id
            }
            user = json.dumps(
                {
                    "reviews": batch_payload,
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
                text = text_by_source_id.get(source_id)
                if text is None:
                    continue
                assertions = item.get("assertions")
                if not isinstance(assertions, list):
                    continue
                validated: list[dict[str, str]] = []
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
                    validated.append(
                        {
                            "aspect": aspect,
                            "subsystem": subsystem,
                            "polarity": polarity,
                            "span": span,
                            "method": "qwen",
                        }
                    )
                extracted_batch[source_id] = validated
            return [(cache_key, extracted_batch[sid]) for sid, cache_key in key_by_source_id.items()]

        if len(batches) == 1:
            batch_results = [_extract_batch(batches[0])]
        else:
            with ThreadPoolExecutor(max_workers=min(4, len(batches))) as executor:
                batch_results = list(executor.map(_extract_batch, batches))

        with _CACHE_LOCK:
            for entries in batch_results:
                for cache_key, validated in entries:
                    _ASPECT_CACHE[cache_key] = validated

    updated = deepcopy(vehicles)
    with _CACHE_LOCK:
        for vehicle, target_keys in zip(updated, target_keys_by_vehicle):
            for review in vehicle.get("reviews", []):
                cache_key = (model_name, str(review["source_id"]), str(review["text"]))
                if cache_key not in target_keys or cache_key not in _ASPECT_CACHE:
                    continue
                assertions = deepcopy(_ASPECT_CACHE[cache_key])
                review["aspect_assertions"] = assertions
                review["aspects"] = _laplace_review_aspects(assertions)
                review["aspect_method"] = "qwen"
    return updated


def _extract_summary_item(
    payload: dict[str, Any],
    candidate_id: str,
) -> dict[str, Any] | None:
    for key in ("candidates", "vehicles", "summaries"):
        entries = payload.get(key)
        if isinstance(entries, list):
            dicts = [item for item in entries if isinstance(item, dict)]
            for item in dicts:
                if str(item.get("candidate_id", "")) == candidate_id:
                    return item
            if len(dicts) == 1:
                return dicts[0]
            return None
    if "strengths" in payload or "cautions" in payload:
        return payload
    raise QwenResponseError("Qwen summary response is missing a candidates list")


def apply_grounded_summaries(
    client: Any,
    candidates: list[dict[str, Any]],
    include_aspect_assertions: bool = True,
) -> list[dict[str, Any]]:
    if not candidates:
        return candidates

    system = (
        "Write short used-car evidence summaries as JSON using the supplied graph paths "
        "to connect related records. Every statement must include "
        "one or more supports containing a supplied evidence ID and an exact quote from "
        "that evidence. Do not create or modify prices, mileage, ratings, "
        "recall counts, fitment, or repair costs. Do not present indirect evidence as an "
        "exact-year fact. Return no more than two strengths and two cautions per vehicle."
    )
    output_schema = {
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
    }

    def _summarize_single(
        candidate: dict[str, Any],
    ) -> tuple[dict[str, Any], bool, Exception | None]:
        updated = deepcopy(candidate)
        template_pros = deepcopy(updated["pros"])
        template_cautions = deepcopy(updated["cautions"])
        evidence_by_id = {
            evidence["id"]: evidence for evidence in updated["evidence"]
        }
        strength_ids = {
            evidence["id"]
            for evidence in updated["evidence"]
            if evidence["type"] in {"listing", "safety", "review"}
            and not evidence["indirect"]
        }
        all_ids = set(evidence_by_id)
        context_evidence = _summary_evidence(updated)
        record = _summary_record(
            updated,
            context_evidence,
            include_aspect_assertions=include_aspect_assertions,
        )
        user = json.dumps({"candidates": [record], "output": output_schema})

        item: dict[str, Any] | None = None
        last_error: Exception | None = None
        responded = False
        try:
            payload = client.complete_json(system, user, max_tokens=800)
            responded = True
            item = _extract_summary_item(payload, str(updated["id"]))
        except (QwenError, TypeError, ValueError, KeyError) as exc:
            last_error = exc

        if item is not None:
            valid_strengths, stripped_strength = _validated_claims(
                item.get("strengths"), evidence_by_id, strength_ids
            )
            valid_cautions, stripped_caution = _validated_claims(
                item.get("cautions"), evidence_by_id, all_ids
            )
        else:
            valid_strengths, stripped_strength = [], False
            valid_cautions, stripped_caution = [], False

        needs_retry = (
            item is None
            or (not valid_strengths and not valid_cautions)
            or (stripped_strength and not valid_strengths)
            or (stripped_caution and not valid_cautions)
        )
        if needs_retry and (responded or isinstance(last_error, QwenResponseError)):
            retry_record = dict(record)
            retry_record["feedback"] = (
                "Previous claims had missing, uncited, or numerically unsupported statements. "
                "Cite ONLY allowed evidence_ids with exact substring quotes and ONLY numbers "
                "present in supported_facts."
            )
            retry_record["allowed_strength_evidence_ids"] = sorted(strength_ids)
            retry_record["allowed_caution_evidence_ids"] = sorted(all_ids)
            retry_record["supported_facts"] = [
                {
                    "evidence_id": ev["id"],
                    "facts": ev.get("facts", {}),
                    "text": ev["text"],
                }
                for ev in context_evidence
            ]
            retry_user = json.dumps(
                {"candidates": [retry_record], "output": output_schema}
            )
            try:
                retry_payload = client.complete_json(system, retry_user, max_tokens=800)
                responded = True
                retry_item = _extract_summary_item(retry_payload, str(updated["id"]))
                if retry_item is not None:
                    r_strengths, _ = _validated_claims(
                        retry_item.get("strengths"), evidence_by_id, strength_ids
                    )
                    r_cautions, _ = _validated_claims(
                        retry_item.get("cautions"), evidence_by_id, all_ids
                    )
                    if r_strengths:
                        valid_strengths = r_strengths
                    if r_cautions:
                        valid_cautions = r_cautions
            except (QwenError, TypeError, ValueError, KeyError) as exc:
                last_error = exc

        if valid_strengths:
            updated["pros"] = valid_strengths
        if valid_cautions:
            updated["cautions"] = _dedupe_claims(template_cautions + valid_cautions)

        try:
            verify_candidate(updated)
        except VerificationError:
            updated["pros"] = template_pros
            updated["cautions"] = template_cautions
            verify_candidate(updated)

        return updated, responded, last_error

    if len(candidates) == 1:
        outcomes = [_summarize_single(candidates[0])]
    else:
        with ThreadPoolExecutor(max_workers=min(3, len(candidates))) as executor:
            outcomes = list(executor.map(_summarize_single, candidates))

    if not any(responded for _, responded, _ in outcomes):
        first_error = next((err for _, _, err in outcomes if err is not None), None)
        if first_error is not None:
            raise first_error

    return [updated for updated, _, _ in outcomes]


def _summary_record(
    candidate: dict[str, Any],
    context_evidence: list[dict[str, Any]],
    include_aspect_assertions: bool = True,
) -> dict[str, Any]:
    context_ids = {item["id"] for item in context_evidence}
    evidence_entries = []
    for item in context_evidence:
        entry: dict[str, Any] = {
            "id": item["id"],
            "text": item["text"],
            "indirect": item["indirect"],
        }
        if include_aspect_assertions and item.get("aspect_assertions"):
            entry["aspect_assertions"] = [
                {
                    "aspect": assertion["aspect"],
                    "subsystem": assertion.get("subsystem", "unknown"),
                    "polarity": assertion["polarity"],
                    "span": str(assertion.get("span", ""))[:200],
                }
                for assertion in item["aspect_assertions"][:6]
            ]
        evidence_entries.append(entry)

    return {
        "candidate_id": candidate["id"],
        "vehicle": {
            "year": candidate["year"],
            "make": candidate["make"],
            "model": candidate["model"],
            "trim": candidate["trim"],
        },
        "evidence": evidence_entries,
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


def _validated_claims(
    value: Any,
    evidence_by_id: dict[str, dict[str, Any]],
    allowed_ids: set[str],
) -> tuple[list[dict[str, Any]], bool]:
    if not isinstance(value, list):
        return [], False
    candidates = _claims(value, evidence_by_id, allowed_ids)
    had_stripped = len(candidates) < min(len(value), 2)
    verified: list[dict[str, Any]] = []
    for claim in candidates:
        try:
            _verify_numeric_claims(
                claim["text"],
                [evidence_by_id[item_id] for item_id in claim["evidence_ids"]],
            )
        except VerificationError:
            had_stripped = True
            continue
        verified.append(claim)
    return verified, had_stripped


def _summary_evidence(
    candidate: dict[str, Any],
    max_records: int = 6,
    max_chars: int = 1_600,
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
        if name in OPTIONAL_FEATURES:
            if parsed < 0.1 and fallback.get(name, 0.0) <= 0:
                continue
        if parsed > 0:
            weights[name] = min(parsed, 1.0)
    if not weights:
        return fallback
    for base in BASELINE_FEATURES:
        if base not in weights and fallback.get(base, 0.0) > 0:
            weights[base] = fallback[base]
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
            ):
                continue
            evidence_text = str(evidence.get("text") or "")
            quote_cf = quote.casefold()
            text_cf = evidence_text.casefold()
            if quote_cf not in text_cf:
                norm_quote = " ".join(quote_cf.split())
                norm_text = " ".join(text_cf.split())
                if norm_quote not in norm_text:
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
