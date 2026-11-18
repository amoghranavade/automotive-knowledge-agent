from __future__ import annotations

import re
from typing import Any


class VerificationError(ValueError):
    pass


def verify_candidate(candidate: dict[str, Any]) -> None:
    evidence_by_id = {item["id"]: item for item in candidate["evidence"]}
    if not evidence_by_id:
        raise VerificationError("Candidate has no evidence")
    for group in (candidate["pros"], candidate["cautions"]):
        for claim in group:
            if not claim["evidence_ids"]:
                raise VerificationError("Claim has no citation")
            missing = set(claim["evidence_ids"]) - set(evidence_by_id)
            if missing:
                raise VerificationError(f"Unknown evidence IDs: {sorted(missing)}")
            _verify_numeric_claims(
                claim["text"],
                [evidence_by_id[item_id] for item_id in claim["evidence_ids"]],
            )

    scoring_ids = candidate["scoring_evidence_ids"]
    unknown_scoring_ids = set(scoring_ids) - set(evidence_by_id)
    if unknown_scoring_ids:
        raise VerificationError(
            f"Unknown scoring evidence IDs: {sorted(unknown_scoring_ids)}"
        )
    if any(
        evidence_by_id[item_id]["indirect"]
        for item_id in scoring_ids
    ):
        raise VerificationError("Indirect evidence cannot affect ranking")

    listing = _single_scoring_evidence(evidence_by_id, scoring_ids, "listing")
    _require_equal(candidate["price"], listing, "price")
    _require_equal(candidate["mileage"], listing, "mileage")

    safety = _single_scoring_evidence(evidence_by_id, scoring_ids, "safety")
    _require_equal(candidate["safety_rating"], safety, "ncap_stars")
    _require_equal(candidate["recall_count"], safety, "recall_count")
    expected_safety_score = (
        candidate["safety_rating"] / 5
        if candidate["safety_rating"] is not None
        else None
    )
    if candidate["score_components"].get("safety") != expected_safety_score:
        raise VerificationError("Safety score does not match the cited rating")

    recall_count = sum(item["type"] == "recall" for item in candidate["evidence"])
    if candidate["recall_count"] != recall_count:
        raise VerificationError("Recall count does not match recall evidence")


def _single_scoring_evidence(
    evidence_by_id: dict[str, dict[str, Any]],
    scoring_ids: list[str],
    evidence_type: str,
) -> dict[str, Any]:
    matches = [
        evidence_by_id[item_id]
        for item_id in scoring_ids
        if evidence_by_id[item_id]["type"] == evidence_type
    ]
    if len(matches) != 1:
        raise VerificationError(f"Candidate needs one cited {evidence_type} record")
    return matches[0]


def _require_equal(value: Any, evidence: dict[str, Any], fact: str) -> None:
    if value != evidence.get("facts", {}).get(fact):
        raise VerificationError(f"Candidate {fact} does not match cited evidence")


def _verify_numeric_claims(text: str, evidence: list[dict[str, Any]]) -> None:
    price_pattern = r"\$\s*([\d,]+(?:\.\d+)?)"
    stated_prices = [_number(match) for match in re.findall(price_pattern, text, re.I)]
    if stated_prices:
        exact_prices: set[float] = set()
        decimal_product_prices: set[float] = set()
        for item in evidence:
            item_prices: list[float] = []
            fact_price = item.get("facts", {}).get("price")
            if fact_price is not None:
                item_prices.append(float(fact_price))
            item_prices.extend(
                _number(match)
                for match in re.findall(price_pattern, str(item.get("text") or ""), re.I)
            )
            exact_prices.update(item_prices)
            if item.get("type") != "listing":
                decimal_product_prices.update(
                    val for val in item_prices if not val.is_integer()
                )
        for value in stated_prices:
            if value in exact_prices:
                continue
            if value.is_integer() and any(
                abs(value - fact_val) < 1.0 for fact_val in decimal_product_prices
            ):
                continue
            raise VerificationError("Claim contains an unsupported price value")

    stated_mileages = _extract_mileages(text)
    if stated_mileages:
        supported_mileages = {
            float(item["facts"]["mileage"])
            for item in evidence
            if "mileage" in item.get("facts", {}) and item["facts"]["mileage"] is not None
        }
        for item in evidence:
            supported_mileages.update(_extract_mileages(str(item.get("text") or "")))
        if any(value not in supported_mileages for value in stated_mileages):
            raise VerificationError("Claim contains an unsupported mileage value")

    checks = (
        (r"(\d(?:\.\d+)?)\s*(?:-|\s)?stars?\b", "ncap_stars"),
        (r"(\d+)\s+(?:model-year\s+)?recall", "recall_count"),
    )
    for pattern, fact in checks:
        stated = [_number(match) for match in re.findall(pattern, text, re.I)]
        if not stated:
            continue
        supported = {
            float(item["facts"][fact])
            for item in evidence
            if fact in item.get("facts", {}) and item["facts"][fact] is not None
        }
        for item in evidence:
            supported.update(
                _number(match)
                for match in re.findall(pattern, str(item.get("text") or ""), re.I)
            )
        if fact == "recall_count":
            supported.add(float(sum(item["type"] == "recall" for item in evidence)))
        if any(value not in supported for value in stated):
            raise VerificationError(f"Claim contains an unsupported {fact} value")


def _extract_mileages(text: str) -> list[float]:
    values = [_number(match) for match in re.findall(r"([\d,]+)\s*(?:miles|mi)\b", text, re.I)]
    values.extend(
        float(match) * 1000.0
        for match in re.findall(r"(\d+(?:\.\d+)?)\s*k\s*(?:miles|mi)\b", text, re.I)
    )
    return values


def _number(value: str) -> float:
    return float(value.replace(",", ""))
