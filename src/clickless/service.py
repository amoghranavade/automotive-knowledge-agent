from __future__ import annotations

import inspect
import re
from typing import Any

from .data.repository import load_catalog
from .graph import retrieve_evidence_paths
from .llm import (
    QwenClient,
    QwenError,
    apply_grounded_summaries,
    apply_model_aspects,
    parse_request_with_model,
)
from .models import Constraints
from .parser import build_navigation_plan, parse
from .ranking import aspect_summary, rank_closest_vehicles, rank_vehicles
from .retrieval import retrieve_passages
from .sentiment import apply_vader_aspects
from .store import data_snapshot, load_vehicles, with_live_recalls
from .verifier import verify_candidate


def parse_request(query: str) -> dict[str, Any]:
    constraints, questions = parse(query)
    return {
        "query": query,
        "constraints": constraints.to_dict(),
        "clarification_questions": questions,
        "navigation_plan": build_navigation_plan(constraints, query),
    }


def _evidence(vehicle: dict[str, Any]) -> list[dict[str, Any]]:
    listing = vehicle["listing"]
    safety = vehicle["safety"]
    items = [
        {
            "id": listing["source_id"],
            "type": "listing",
            "source": listing["source"],
            "url": listing["url"],
            "observed_at": listing["observed_at"],
            "text": f"Listed at ${listing['price']:,} with {listing['mileage']:,} miles.",
            "facts": {"price": listing["price"], "mileage": listing["mileage"]},
            "indirect": False,
            "data_mode": listing.get("data_mode", "frozen_snapshot"),
        },
        {
            "id": safety["source_id"],
            "type": "safety",
            "source": "NHTSA",
            "url": safety["url"],
            "observed_at": safety["observed_at"],
            "text": _safety_text(safety),
            "facts": {
                "ncap_stars": safety["ncap_stars"],
                "recall_count": len(safety.get("recalls", [])),
            },
            "indirect": False,
        },
    ]
    for recall in safety.get("recalls", []):
        campaign = recall["campaign_number"] or "campaign number unavailable"
        items.append(
            {
                "id": recall["source_id"],
                "type": "recall",
                "source": recall["source"],
                "url": recall["url"],
                "observed_at": recall["observed_at"],
                "text": f"{campaign}: {recall['component']}. {recall['summary']}",
                "facts": {
                    "campaign_number": recall["campaign_number"],
                    "component": recall["component"],
                },
                "indirect": False,
            }
        )
    exact_reviews = [
        item for item in vehicle.get("reviews", [])
        if item.get("match_type") == "exact_year"
    ]
    indirect_reviews = [
        item for item in vehicle.get("reviews", [])
        if item.get("match_type") != "exact_year"
    ]
    for review in exact_reviews[:6] + indirect_reviews[:2]:
        items.append(
            {
                "id": review["source_id"],
                "type": "review",
                "source": review["source"],
                "url": review["url"],
                "observed_at": review["observed_at"],
                "text": review["text"],
                "facts": {"review_year": review["year"]},
                "aspect_assertions": review.get("aspect_assertions", []),
                "match_type": review.get("match_type", "exact_year"),
                "source_kind": review.get("source_kind", "owner_review"),
                "indirect": review.get("match_type") != "exact_year",
                "data_mode": review.get("data_mode", "frozen_snapshot"),
            }
        )
    for product in vehicle.get("products", [])[:3]:
        items.append(
            {
                "id": product["source_id"],
                "type": "aftermarket_product",
                "source": product["source"],
                "url": product["url"],
                "observed_at": product["observed_at"],
                "text": product["text"],
                "facts": {
                    "verified_fitment": product["verified_fitment"],
                    "price": product.get("price"),
                },
                "indirect": False,
                "data_mode": product.get("data_mode", "frozen_snapshot"),
            }
        )
    return items


def _concise_text(text: str, max_len: int = 240) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= max_len:
        return cleaned
    cutoff = cleaned.rfind(" ", 0, max_len - 2)
    if cutoff <= 0:
        cutoff = max_len - 3
    return cleaned[:cutoff].rstrip(" ,;:") + "..."


def _concise_review_pro(
    exact_reviews: list[dict[str, Any]],
    configuration: str,
) -> dict[str, Any] | None:
    if not exact_reviews:
        return None
    if configuration != "D":
        for review in exact_reviews:
            for assertion in review.get("aspect_assertions", []):
                if assertion.get("polarity") == "positive":
                    span = str(assertion.get("span") or "").strip()
                    if span:
                        return {
                            "text": _concise_text(span, 240),
                            "evidence_ids": [review["id"]],
                        }
    for review in exact_reviews:
        assertions = review.get("aspect_assertions", [])
        if assertions and all(a.get("polarity") == "negative" for a in assertions):
            continue
        raw_text = str(review.get("text") or "").strip()
        if raw_text:
            first_sentence = re.split(r"(?<=[.!?])\s+", raw_text, maxsplit=1)[0]
            return {
                "text": _concise_text(first_sentence, 240),
                "evidence_ids": [review["id"]],
            }
    for review in exact_reviews:
        raw_text = str(review.get("text") or "").strip()
        if raw_text:
            first_sentence = re.split(r"(?<=[.!?])\s+", raw_text, maxsplit=1)[0]
            return {
                "text": _concise_text(first_sentence, 240),
                "evidence_ids": [review["id"]],
            }
    return None


def _candidate(
    vehicle: dict[str, Any],
    score: float,
    components: dict[str, float | None],
    missing: list[str],
    constraint_violations: list[str] | None = None,
    configuration: str = "C",
) -> dict[str, Any]:
    evidence = _evidence(vehicle)
    exact_reviews = [item for item in evidence if item["type"] == "review" and not item["indirect"]]
    listing_id = vehicle["listing"]["source_id"]
    safety_id = vehicle["safety"]["source_id"]
    stars = vehicle["safety"]["ncap_stars"]
    safety_text = _safety_text(vehicle["safety"])
    pros = [{"text": safety_text, "evidence_ids": [safety_id]}] if stars else []
    review_pro = _concise_review_pro(exact_reviews, configuration)
    if review_pro is not None:
        pros.append(review_pro)
    recall_items = [item for item in evidence if item["type"] == "recall"]
    if recall_items:
        source_label = "NHTSA returned" if vehicle["safety"]["recall_mode"] == "live" else "The NHTSA snapshot contains"
        cautions = [
            {
                "text": f"{source_label} {len(recall_items)} model-year recall caution(s); verify completion for the specific VIN.",
                "evidence_ids": [safety_id] + [item["id"] for item in recall_items],
            }
        ]
    else:
        cautions = [
            {
                "text": "No recall record is stored in the NHTSA snapshot; verify the VIN directly with NHTSA.",
                "evidence_ids": [safety_id],
            }
        ]
    indirect_reviews = [
        item
        for item in evidence
        if item["type"] == "review"
        and item["indirect"]
        and item.get("match_type", "same_generation") == "same_generation"
        and item.get("source_kind", "owner_review") != "product_review"
    ]
    if indirect_reviews:
        cautions.append(
            {
                "text": "Sibling-year review evidence is indirect and does not affect the score.",
                "evidence_ids": [item["id"] for item in indirect_reviews],
            }
        )
    related_products = [
        item for item in evidence if item["type"] == "aftermarket_product"
    ]
    if related_products:
        cautions.append(
            {
                "text": "Related products do not prove fitment or total repair cost.",
                "evidence_ids": [item["id"] for item in related_products],
            }
        )
    candidate = {
        "aspect_bundles": (
            _aspect_bundles(vehicle, evidence) if configuration in {"C", "E"} else []
        ),
        "id": vehicle["id"],
        "year": vehicle["year"],
        "make": vehicle["make"],
        "model": vehicle["model"],
        "trim": vehicle["trim"],
        "body_style": vehicle["body_style"],
        "drivetrain": vehicle["drivetrain"],
        "price": vehicle["listing"]["price"],
        "mileage": vehicle["listing"]["mileage"],
        "location": vehicle["listing"].get("location", ""),
        "listing_source": vehicle["listing"].get("source"),
        "listing_data_mode": vehicle["listing"].get("data_mode", "frozen_snapshot"),
        "safety_rating": stars,
        "safety_basis": vehicle["safety"].get("rating_basis", "overall"),
        "recall_count": len(recall_items),
        "score": score,
        "tier": "requires_additional_evidence" if missing else "complete",
        "score_components": components,
        "aspect_summary": aspect_summary(vehicle),
        "missing_evidence": missing,
        "constraint_violations": constraint_violations or [],
        "pros": pros,
        "cautions": cautions,
        "evidence": evidence,
        "scoring_evidence_ids": [listing_id, safety_id]
        + [item["id"] for item in exact_reviews],
    }
    verify_candidate(candidate)
    return candidate


def _retrieve_graph_paths(
    catalog: dict[str, Any],
    vehicle_id: str,
    aspects: list[str] | None,
    live_recalls: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    params = inspect.signature(retrieve_evidence_paths).parameters
    kwargs: dict[str, Any] = {}
    if "aspects" in params:
        kwargs["aspects"] = aspects
    if "live_recalls" in params:
        kwargs["live_recalls"] = live_recalls
    return retrieve_evidence_paths(catalog, vehicle_id, **kwargs)


def _live_recalls_payload(vehicle: dict[str, Any]) -> list[dict[str, Any]] | None:
    safety = vehicle.get("safety", {})
    if safety.get("recall_mode") != "live":
        return None
    payload = []
    for index, recall in enumerate(safety.get("recalls", [])):
        source_id = str(recall.get("source_id") or f"nhtsa-live-{index}")
        payload.append(
            {
                "id": f"{vehicle['id']}-{source_id}-{index}",
                "vehicle_id": vehicle["id"],
                "campaign_number": recall.get("campaign_number", ""),
                "component": recall.get("component", "unknown"),
                "summary": recall.get("summary", ""),
                "remedy": recall.get("remedy", ""),
                "source_id": source_id,
                "source_name": recall.get("source", "NHTSA"),
                "source_url": recall.get("url", "https://www.nhtsa.gov/recalls"),
                "captured_at": recall.get("observed_at", "unknown"),
            }
        )
    return payload


def recommend(
    query: str,
    live_nhtsa: bool = False,
    use_model: bool = True,
    configuration: str = "C",
    candidate_limit: int = 3,
) -> dict[str, Any]:
    configuration = configuration.upper()
    if configuration not in {"A", "B", "C", "D", "E"}:
        raise ValueError("Configuration must be A, B, C, D, or E")
    if not 1 <= candidate_limit <= 10:
        raise ValueError("candidate_limit must be between 1 and 10")
    constraints, questions = parse(query)
    client = None
    model_notices = []
    model_stages = {
        "parsing": "rules",
        "review_extraction": "stored_lexicon",
        "summaries": "templates",
    }
    if use_model and configuration != "A":
        try:
            client = QwenClient.from_env()
            constraints, questions = parse_request_with_model(
                client,
                query,
                constraints,
                questions,
            )
            model_stages["parsing"] = "qwen"
        except (QwenError, TypeError, ValueError, KeyError) as error:
            client = None
            model_notices.append(f"Qwen unavailable; used rule-based parsing. {error}")

    parsed_constraints = constraints
    navigation_plan = build_navigation_plan(parsed_constraints, query)
    ranking_constraints = (
        _without_sentiment_weights(parsed_constraints)
        if configuration == "E"
        else parsed_constraints
    )

    vehicles = load_vehicles(live_nhtsa=False)
    if configuration == "A":
        try:
            vehicles = apply_vader_aspects(vehicles)
            model_stages["review_extraction"] = "vader"
        except RuntimeError as error:
            model_notices.append(str(error))

    if client is not None and configuration in {"B", "C", "D", "E"}:
        preliminary = rank_vehicles(vehicles, parsed_constraints)
        if not preliminary:
            preliminary = rank_closest_vehicles(vehicles, parsed_constraints)
        pool_size = min(max(candidate_limit + 1, 4), 5)
        extraction_pool = [item[0] for item in preliminary[:pool_size]]
        if configuration == "E":
            preliminary_e = rank_vehicles(vehicles, ranking_constraints)
            if not preliminary_e:
                preliminary_e = rank_closest_vehicles(vehicles, ranking_constraints)
            seen_ids = {item["id"] for item in extraction_pool}
            for item in preliminary_e[:pool_size]:
                if item[0]["id"] not in seen_ids:
                    extraction_pool.append(item[0])
                    seen_ids.add(item[0]["id"])
        try:
            enriched = apply_model_aspects(
                client,
                extraction_pool,
                max_reviews_per_vehicle=2,
            )
            by_id = {vehicle["id"]: vehicle for vehicle in enriched}
            vehicles = [by_id.get(vehicle["id"], vehicle) for vehicle in vehicles]
            model_stages["review_extraction"] = "qwen"
        except (QwenError, TypeError, ValueError, KeyError) as error:
            model_notices.append(f"Review aspects used stored evidence. {error}")

    ranked = rank_vehicles(vehicles, ranking_constraints)
    feasible_count = len(ranked)
    comparable_cohort_count = sum(1 for _, _, _, missing in ranked if not missing)
    cohort_retention = (
        round(comparable_cohort_count / feasible_count, 4) if feasible_count else None
    )
    match_mode = "exact"
    selected_vehicles: list[dict[str, Any]] = []
    if ranked:
        ranked = _top_distinct(ranked, candidate_limit)
        candidates = []
        for vehicle, score, components, missing in ranked:
            final_vehicle = with_live_recalls(vehicle) if live_nhtsa else vehicle
            selected_vehicles.append(final_vehicle)
            candidates.append(
                _candidate(
                    final_vehicle,
                    score,
                    components,
                    missing,
                    configuration=configuration,
                )
            )
    else:
        match_mode = "closest_alternatives"
        closest = _top_distinct(
            rank_closest_vehicles(vehicles, ranking_constraints),
            candidate_limit,
        )
        candidates = []
        for vehicle, score, components, missing, violations in closest:
            final_vehicle = with_live_recalls(vehicle) if live_nhtsa else vehicle
            selected_vehicles.append(final_vehicle)
            candidates.append(
                _candidate(
                    final_vehicle,
                    score,
                    components,
                    missing,
                    violations,
                    configuration=configuration,
                )
            )

    catalog = load_catalog()
    requested_aspects = [
        name
        for name in ("reliability", "maintenance", "fuel_economy", "cargo")
        if parsed_constraints.preferences.get(name, 0) > 0
    ] or None

    for selected_vehicle, candidate in zip(selected_vehicles, candidates):
        if configuration == "A":
            candidate["retrieved_passages"] = retrieve_passages(
                query, candidate["evidence"], "bm25"
            )
            candidate["graph_paths"] = []
            candidate["retrieval_mode"] = "bm25"
        elif configuration == "B":
            try:
                candidate["retrieved_passages"] = retrieve_passages(
                    query, candidate["evidence"], "dense"
                )
                candidate["retrieval_mode"] = "all-MiniLM-L6-v2"
            except RuntimeError as error:
                candidate["retrieved_passages"] = retrieve_passages(
                    query, candidate["evidence"], "tfidf"
                )
                candidate["retrieval_mode"] = "tfidf_fallback"
                model_notices.append(str(error))
            candidate["graph_paths"] = []
        else:
            live_payload = (
                _live_recalls_payload(selected_vehicle) if live_nhtsa else None
            )
            graph_paths = _retrieve_graph_paths(
                catalog,
                candidate["id"],
                requested_aspects,
                live_payload,
            )
            backend = graph_paths[0].get("backend", "memory") if graph_paths else "memory"
            if configuration == "D":
                graph_paths = [
                    path
                    for path in graph_paths
                    if path["kind"] != "review"
                    and "review" not in path.get("source_kinds", [])
                ]
                graph_paths.extend(_raw_review_paths(candidate, backend))
            existing_lineage_ids = {
                evidence_id
                for path in graph_paths
                if path.get("kind") == "generation_lineage"
                for evidence_id in path.get("evidence_ids", [])
            }
            for path in _lineage_paths(candidate, backend):
                if not any(
                    eid in existing_lineage_ids
                    for eid in path.get("evidence_ids", [])
                ):
                    graph_paths.append(path)
            candidate["graph_paths"] = graph_paths
            try:
                candidate["retrieved_passages"] = retrieve_passages(
                    query, candidate["evidence"], "dense"
                )
                dense_mode = "all-MiniLM-L6-v2"
            except RuntimeError as error:
                candidate["retrieved_passages"] = retrieve_passages(
                    query, candidate["evidence"], "tfidf"
                )
                dense_mode = "tfidf_fallback"
                model_notices.append(str(error))
            candidate["retrieval_mode"] = f"ontology_graph+{dense_mode}"
    if client is not None:
        try:
            candidates = apply_grounded_summaries(
                client,
                candidates,
                include_aspect_assertions=(configuration != "D"),
            )
            model_stages["summaries"] = "qwen"
        except (QwenError, TypeError, ValueError, KeyError) as error:
            model_notices.append(f"Recommendation summaries used stored evidence. {error}")
    status = "ok" if match_mode == "exact" else "closest_alternatives"
    if questions:
        status = "needs_clarification"
    snapshot = data_snapshot()
    used_stages = sum(value == "qwen" for value in model_stages.values())
    model_mode = "rules"
    if model_stages["parsing"] == "qwen" and model_stages["summaries"] == "qwen":
        model_mode = "qwen"
    elif used_stages:
        model_mode = "qwen_partial"
    return {
        "query": query,
        "parsed": ranking_constraints.to_dict(),
        "clarification_questions": questions,
        "navigation_plan": navigation_plan,
        "candidates": candidates,
        "status": status,
        "match_mode": match_mode,
        "message": (
            "No exact matches were found. Showing the closest alternatives with unmet constraints labeled."
            if match_mode == "closest_alternatives"
            else None
        ),
        "feasible_count": feasible_count,
        "comparable_cohort_count": comparable_cohort_count,
        "cohort_retention": cohort_retention,
        "data_mode": snapshot.get("mode", "frozen_snapshot"),
        "data_notice": snapshot.get("notice"),
        "model_mode": model_mode,
        "model_name": client.model if client is not None else None,
        "model_notice": " ".join(dict.fromkeys(model_notices)) or None,
        "abstained": match_mode != "exact",
        "cohort": {
            "feasible": feasible_count,
            "fully_scored": comparable_cohort_count,
            "retention": cohort_retention,
        },
        "model_stages": model_stages,
        "configuration": configuration,
    }


def _without_sentiment_weights(constraints: Constraints) -> Constraints:
    remaining = {
        name: weight
        for name, weight in constraints.preferences.items()
        if name not in {"reliability", "maintenance", "fuel_economy", "cargo"}
    }
    total = sum(remaining.values()) or 1.0
    return Constraints(
        max_price=constraints.max_price,
        max_mileage=constraints.max_mileage,
        min_year=constraints.min_year,
        body_style=constraints.body_style,
        drivetrain=constraints.drivetrain,
        make=constraints.make,
        preferences={name: round(weight / total, 4) for name, weight in remaining.items()},
    )


def _top_distinct(ranked: list[tuple[Any, ...]], limit: int) -> list[tuple[Any, ...]]:
    selected = []
    seen = set()
    for item in ranked:
        vehicle = item[0]
        key = (vehicle["year"], vehicle["make"].casefold(), vehicle["model"].casefold())
        if key in seen:
            continue
        seen.add(key)
        selected.append(item)
        if len(selected) == limit:
            break
    return selected


def _raw_review_paths(candidate: dict[str, Any], backend: str) -> list[dict[str, Any]]:
    return [
        {
            "kind": "review",
            "label": "Owner review (raw text)",
            "component": None,
            "node_ids": [candidate["id"], item["id"]],
            "relations": ["HAS_REVIEW"],
            "evidence_ids": [item["id"]],
            "text": "Owner review retrieved without aspect-assertion nodes.",
            "indirect": item["indirect"],
            "backend": backend,
        }
        for item in candidate["evidence"]
        if item["type"] == "review"
    ][:4]


def _lineage_paths(candidate: dict[str, Any], backend: str) -> list[dict[str, Any]]:
    return [
        {
            "kind": "generation_lineage",
            "label": "Same-generation owner review",
            "component": None,
            "node_ids": [candidate["id"], item["id"]],
            "relations": ["SHARES_GENERATION", "HAS_REVIEW"],
            "evidence_ids": [item["id"]],
            "text": "Sibling-variant review shown as indirect lineage evidence and excluded from ranking.",
            "indirect": True,
            "backend": backend,
        }
        for item in candidate["evidence"]
        if item["type"] == "review"
        and item["indirect"]
        and item.get("match_type", "same_generation") == "same_generation"
        and item.get("source_kind", "owner_review") != "product_review"
    ][:2]


ASPECT_NAMES = ("reliability", "maintenance", "fuel_economy", "cargo")


def _aspect_bundles(
    vehicle: dict[str, Any], evidence: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Slide 6, query 3: one supporting and one opposing span per aspect."""
    cited = {item["id"] for item in evidence if item["type"] == "review" and not item["indirect"]}
    reviews = [
        review
        for review in vehicle.get("reviews", [])
        if review.get("match_type") == "exact_year" and review["source_id"] in cited
    ]
    qwen = [review for review in reviews if review.get("aspect_method") == "qwen"]
    reviews = qwen or reviews
    bundles = []
    for aspect in ASPECT_NAMES:
        bundle: dict[str, Any] = {"aspect": aspect, "positive": None, "negative": None}
        for review in reviews:
            for assertion in review.get("aspect_assertions", []):
                if assertion.get("aspect") != aspect:
                    continue
                side = assertion.get("polarity")
                if side in {"positive", "negative"} and bundle[side] is None:
                    bundle[side] = {
                        "text": str(assertion.get("span") or "")[:240],
                        "evidence_ids": [review["source_id"]],
                    }
        if bundle["positive"] or bundle["negative"]:
            bundles.append(bundle)
    return bundles


def _safety_text(safety: dict[str, Any]) -> str:
    stars = safety.get("ncap_stars")
    if stars is None:
        return "NHTSA has no crash-test scores for this model year in the stored data."
    if safety.get("rating_basis") == "test_average":
        return (
            f"NHTSA crash tests average {stars} stars (front, side, rollover); "
            "this model year has no overall rating."
        )
    return f"NHTSA reports an overall {stars}-star rating."
