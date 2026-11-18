from __future__ import annotations

from copy import deepcopy
from typing import Any

from .data.nhtsa import NHTSAError, fetch_recalls, recalls_with_fallback
from .data.repository import all_vehicle_bundles, load_catalog


POLARITY_SCORE = {
    "positive": 1.0,
    "negative": 0.0,
    "neutral": 0.5,
    "mixed": 0.5,
}


def load_vehicles(live_nhtsa: bool = False) -> list[dict[str, Any]]:
    catalog = load_catalog()
    bundles = all_vehicle_bundles(catalog)
    vehicle_by_id = {item["id"]: item for item in catalog["vehicles"]}
    reviews_by_generation: dict[str, list[dict[str, Any]]] = {}
    for bundle in bundles:
        generation = bundle["vehicle"]["generation_id"]
        reviews_by_generation.setdefault(generation, []).extend(bundle["reviews"])
    vehicles = []
    for bundle in bundles:
        vehicle = bundle["vehicle"]
        if not bundle["listings"]:
            continue

        listings = [
            _listing(item)
            for item in sorted(
                bundle["listings"],
                key=lambda r: (r.get("data_mode") != "live", r["price"], r["mileage"]),
            )
        ]
        safety_record = bundle["safety_ratings"][0] if bundle["safety_ratings"] else None
        recalls, recall_mode = recalls_with_fallback(
            catalog,
            vehicle,
            live=live_nhtsa,
        )

        exact_owner_reviews = [
            item
            for item in bundle["reviews"]
            if item.get("source_kind", "owner_review") == "owner_review"
            and item.get("match_type", "exact_year") == "exact_year"
        ]
        indirect_reviews = []
        if len(exact_owner_reviews) < 5:
            for item in reviews_by_generation.get(vehicle["generation_id"], []):
                source_vehicle = vehicle_by_id.get(item["vehicle_id"])
                if item["vehicle_id"] == vehicle["id"] or source_vehicle is None:
                    continue
                if item.get("source_kind", "owner_review") != "owner_review":
                    continue
                indirect = dict(item)
                indirect["match_type"] = "same_generation"
                indirect_reviews.append(indirect)
                if len(indirect_reviews) == 5:
                    break

        vehicles.append(
            {
                "id": vehicle["id"],
                "year": vehicle["year"],
                "make": vehicle["make"],
                "model": vehicle["model"],
                "trim": vehicle["trim"],
                "body_style": vehicle["body_type"].lower(),
                "drivetrain": vehicle["drivetrain"],
                "listing": listings[0],
                "listings": listings,
                "safety": _safety(safety_record, recalls, recall_mode),
                "reviews": [
                    _review(item) for item in bundle["reviews"] + indirect_reviews
                ],
                "products": [_product(item) for item in bundle["parts"]],
            }
        )
    return vehicles


def data_snapshot() -> dict[str, Any]:
    return load_catalog()["snapshot"]


def with_live_recalls(vehicle: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(vehicle)
    try:
        records = fetch_recalls(vehicle["make"], vehicle["model"], vehicle["year"])
    except NHTSAError:
        return updated
    seen_ids: dict[str, int] = {}
    recalls = []
    for item in records:
        recall = _recall(item)
        source_id = recall["source_id"]
        count = seen_ids.get(source_id, 0)
        seen_ids[source_id] = count + 1
        if count:
            recall["source_id"] = f"{source_id}-{count}"
        recalls.append(recall)
    updated["safety"]["recalls"] = recalls
    updated["safety"]["recall_count"] = len(recalls)
    updated["safety"]["recall_mode"] = "live"
    return updated


def _listing(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "price": record["price"],
        "mileage": record["mileage"],
        "source": record["source_name"],
        "source_id": record["source_id"],
        "url": record["source_url"],
        "observed_at": record["captured_at"],
        "location": record.get("location", ""),
        "data_mode": record.get("data_mode", "frozen_snapshot"),
    }


def _safety(
    record: dict[str, Any] | None,
    recalls: list[dict[str, Any]],
    recall_mode: str,
) -> dict[str, Any]:
    if record is None:
        return {
            "ncap_stars": None,
            "recall_count": len(recalls),
            "source_id": "nhtsa-unavailable",
            "source": "NHTSA",
            "url": "https://www.nhtsa.gov/ratings",
            "observed_at": "unknown",
            "recall_mode": recall_mode,
            "recalls": [_recall(item) for item in recalls],
        }
    return {
        "ncap_stars": record["overall_stars"],
        "recall_count": len(recalls),
        "source_id": record["source_id"],
        "source": record["source_name"],
        "url": record["source_url"],
        "observed_at": record["captured_at"],
        "recall_mode": recall_mode,
        "recalls": [_recall(item) for item in recalls],
    }


def _recall(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_id": record.get("source_id") or record["id"],
        "campaign_number": record.get("campaign_number", ""),
        "component": record.get("component", "unknown"),
        "summary": record.get("summary", ""),
        "remedy": record.get("remedy", ""),
        "source": record.get("source_name", "NHTSA"),
        "url": record.get("source_url", "https://www.nhtsa.gov/recalls"),
        "observed_at": record.get("captured_at") or record.get("report_received_date", "unknown"),
    }


def _review(record: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {}
    for assertion in record.get("aspects", []):
        grouped.setdefault(assertion["aspect"], []).append(
            POLARITY_SCORE[assertion["polarity"]]
        )
    aspects = {
        name: round(sum(values) / len(values), 4)
        for name, values in grouped.items()
    }
    return {
        "year": record["review_year"],
        "source": record["source_name"],
        "source_id": record["source_id"],
        "url": record["source_url"],
        "observed_at": record["captured_at"],
        "data_mode": record.get("data_mode", "frozen_snapshot"),
        "text": record["text"],
        "aspects": aspects,
        "aspect_assertions": [
            {
                "aspect": assertion["aspect"],
                "subsystem": assertion.get("subsystem", "unknown"),
                "polarity": assertion["polarity"],
                "span": assertion.get("span", ""),
                "method": assertion.get("method", "lexicon"),
            }
            for assertion in record.get("aspects", [])
        ],
        "aspect_method": "lexicon",
        "match_type": record.get("match_type", "exact_year"),
        "source_kind": record.get("source_kind", "owner_review"),
    }


def _product(record: dict[str, Any]) -> dict[str, Any]:
    fitment = "fitment verified" if record["verified_fitment"] else "fitment not verified"
    return {
        "source": record["source_name"],
        "source_id": record["source_id"],
        "url": record["source_url"],
        "observed_at": record["captured_at"],
        "data_mode": record.get("data_mode", "frozen_snapshot"),
        "price": record["price"],
        "text": f"{record['name']} listed at ${record['price']:,.2f}; {fitment}.",
        "verified_fitment": record["verified_fitment"],
    }
