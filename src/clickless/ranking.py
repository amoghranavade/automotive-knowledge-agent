from __future__ import annotations

from typing import Any

from .models import Constraints


CANONICAL_ASPECTS = ("reliability", "maintenance", "fuel_economy", "cargo")


# Many listings (Craigslist especially) label all-wheel-drive SUVs "4WD", so an
# AWD request also accepts 4WD vehicles. Every other drivetrain must match exactly.
DRIVETRAIN_EQUIVALENTS = {"AWD": {"AWD", "4WD"}}


def drivetrain_matches(actual: Any, wanted: Any) -> bool:
    if not wanted:
        return True
    actual_key = str(actual or "").upper()
    wanted_key = str(wanted).upper()
    return actual_key in DRIVETRAIN_EQUIVALENTS.get(wanted_key, {wanted_key})


def select_best_listing(
    vehicle: dict[str, Any],
    constraints: Constraints,
    allow_violations: bool = False,
) -> dict[str, Any] | None:
    """Choose the highest-scoring compliant listing, or closest fallback if allowed."""
    listings = list(vehicle.get("listings") or ())
    if not listings and vehicle.get("listing") is not None:
        listings = [vehicle["listing"]]
    if not listings:
        return None

    compliant = [
        item
        for item in listings
        if (constraints.max_price is None or item["price"] <= constraints.max_price)
        and (constraints.max_mileage is None or item["mileage"] <= constraints.max_mileage)
    ]
    if compliant:
        price_ceiling = constraints.max_price or max(
            max(item["price"] for item in compliant) * 1.25,
            1,
        )
        mileage_ceiling = constraints.max_mileage or max(
            120_000,
            max(item["mileage"] for item in compliant),
        )
        w_price = constraints.preferences.get("price", 0.30)
        w_mileage = constraints.preferences.get("mileage", 0.20)
        if w_price + w_mileage <= 0:
            w_price, w_mileage = 0.5, 0.5

        def compliant_key(item: dict[str, Any]) -> tuple[bool, float, int, int]:
            p_score = max(0.0, 1.0 - item["price"] / price_ceiling)
            m_score = max(0.0, 1.0 - item["mileage"] / mileage_ceiling)
            combined = w_price * p_score + w_mileage * m_score
            return (
                item.get("data_mode") != "live",
                -combined,
                item["price"],
                item["mileage"],
            )

        return min(compliant, key=compliant_key)

    if not allow_violations:
        return None

    def violation_key(item: dict[str, Any]) -> tuple[int, float, int, int, bool]:
        viol_count = 0
        penalty = 0.0
        if constraints.max_price is not None and item["price"] > constraints.max_price:
            viol_count += 1
            penalty += (item["price"] - constraints.max_price) / max(constraints.max_price, 1)
        if constraints.max_mileage is not None and item["mileage"] > constraints.max_mileage:
            viol_count += 1
            penalty += (item["mileage"] - constraints.max_mileage) / max(constraints.max_mileage, 1)
        return (
            viol_count,
            penalty,
            item["price"],
            item["mileage"],
            item.get("data_mode") != "live",
        )

    return min(listings, key=violation_key)


def filter_vehicles(
    vehicles: list[dict[str, Any]], constraints: Constraints
) -> list[dict[str, Any]]:
    matches = []
    for vehicle in vehicles:
        if constraints.min_year is not None and vehicle["year"] < constraints.min_year:
            continue
        if constraints.body_style and vehicle["body_style"] != constraints.body_style:
            continue
        if not drivetrain_matches(vehicle["drivetrain"], constraints.drivetrain):
            continue
        if (
            constraints.make
            and str(vehicle.get("make", "")).casefold() != constraints.make.casefold()
        ):
            continue
        best_listing = select_best_listing(vehicle, constraints, allow_violations=False)
        if best_listing is None:
            continue
        matches.append({**vehicle, "listing": best_listing})
    return matches


def _eligible_reviews(vehicle: dict[str, Any], aspect_name: str) -> list[dict[str, Any]]:
    return [
        review
        for review in vehicle.get("reviews", [])
        if review.get("match_type") == "exact_year"
        or (
            aspect_name == "maintenance"
            and review.get("source_kind") == "product_review"
        )
    ]


def _aspect_counts(vehicle: dict[str, Any], name: str) -> dict[str, Any]:
    eligible = _eligible_reviews(vehicle, name)
    qwen_reviews = [review for review in eligible if review.get("aspect_method") == "qwen"]
    qwen_assertions = [
        assertion
        for review in qwen_reviews
        for assertion in review.get("aspect_assertions", [])
        if assertion.get("aspect") == name
    ]
    if qwen_assertions:
        assertions = qwen_assertions
    else:
        assertions = [
            assertion
            for review in eligible
            for assertion in review.get("aspect_assertions", [])
            if assertion.get("aspect") == name
        ]

    n_pos = sum(1 for item in assertions if item.get("polarity") == "positive")
    n_neg = sum(1 for item in assertions if item.get("polarity") == "negative")
    n_neu = sum(1 for item in assertions if item.get("polarity") == "neutral")
    n_mix = sum(1 for item in assertions if item.get("polarity") == "mixed")
    total = n_pos + n_neg + n_neu + n_mix
    if total == 0:
        return {
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "mixed": 0,
            "total": 0,
            "net_polarity": None,
            "score": None,
        }
    net_polarity = (n_pos - n_neg) / (total + 1)
    score = (net_polarity + 1.0) / 2.0
    return {
        "positive": n_pos,
        "negative": n_neg,
        "neutral": n_neu,
        "mixed": n_mix,
        "total": total,
        "net_polarity": round(net_polarity, 4),
        "score": round(score, 4),
    }


def aspect_summary(vehicle: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Summarize assertion counts and Laplace-smoothed polarity scores per canonical aspect."""
    return {name: _aspect_counts(vehicle, name) for name in CANONICAL_ASPECTS}


def _aspect(vehicle: dict[str, Any], name: str) -> float | None:
    summary = _aspect_counts(vehicle, name)
    if summary["total"] > 0:
        return summary["score"]
    eligible = _eligible_reviews(vehicle, name)
    if any("aspect_assertions" in review for review in eligible):
        return None
    qwen_values = [
        review["aspects"][name]
        for review in eligible
        if review.get("aspect_method") == "qwen" and name in review.get("aspects", {})
    ]
    values = qwen_values or [
        review["aspects"][name]
        for review in eligible
        if name in review.get("aspects", {})
    ]
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def score_vehicle(
    vehicle: dict[str, Any], constraints: Constraints
) -> tuple[float, dict[str, float | None], list[str]]:
    listing = (
        select_best_listing(vehicle, constraints, allow_violations=True)
        or vehicle["listing"]
    )
    price_ceiling = constraints.max_price or max(listing["price"] * 1.25, 1)
    mileage_ceiling = constraints.max_mileage or 120_000
    components: dict[str, float | None] = {
        "price": max(0.0, 1.0 - listing["price"] / price_ceiling),
        "mileage": max(0.0, 1.0 - listing["mileage"] / mileage_ceiling),
        "safety": vehicle["safety"].get("ncap_stars", 0) / 5
        if vehicle.get("safety", {}).get("ncap_stars") is not None
        else None,
        "reliability": _aspect(vehicle, "reliability"),
        "maintenance": _aspect(vehicle, "maintenance"),
        "fuel_economy": _aspect(vehicle, "fuel_economy"),
        "cargo": _aspect(vehicle, "cargo"),
        "snow": {
            "AWD": 1.0,
            "4WD": 1.0,
            "FWD": 0.55,
            "RWD": 0.25,
        }.get(vehicle["drivetrain"], 0.4),
    }
    missing = [
        name
        for name, weight in constraints.preferences.items()
        if weight > 0 and components.get(name) is None
    ]
    available_weight = sum(
        weight
        for name, weight in constraints.preferences.items()
        if components.get(name) is not None
    )
    score = 0.0
    if available_weight:
        score = sum(
            constraints.preferences[name] * float(components[name])
            for name in constraints.preferences
            if components.get(name) is not None
        ) / available_weight
    return round(score, 4), components, missing


def rank_vehicles(
    vehicles: list[dict[str, Any]], constraints: Constraints
) -> list[tuple[dict[str, Any], float, dict[str, float | None], list[str]]]:
    scored = [
        (vehicle, *score_vehicle(vehicle, constraints))
        for vehicle in filter_vehicles(vehicles, constraints)
    ]
    return sorted(scored, key=lambda item: (bool(item[3]), -item[1]))


def constraint_violations(
    vehicle: dict[str, Any], constraints: Constraints
) -> tuple[list[str], float]:
    listing = (
        select_best_listing(vehicle, constraints, allow_violations=True)
        or vehicle["listing"]
    )
    violations = []
    penalty = 0.0
    if constraints.max_price is not None and listing["price"] > constraints.max_price:
        excess = listing["price"] - constraints.max_price
        violations.append(f"Price is ${excess:,} above the ${constraints.max_price:,} limit.")
        penalty += excess / constraints.max_price
    if constraints.max_mileage is not None and listing["mileage"] > constraints.max_mileage:
        excess = listing["mileage"] - constraints.max_mileage
        violations.append(f"Mileage is {excess:,} miles above the requested limit.")
        penalty += excess / constraints.max_mileage
    if constraints.min_year is not None and vehicle["year"] < constraints.min_year:
        difference = constraints.min_year - vehicle["year"]
        violations.append(f"Model year is {difference} year(s) older than requested.")
        penalty += difference * 0.05
    if constraints.body_style and vehicle["body_style"] != constraints.body_style:
        violations.append(
            f"Body style is {vehicle['body_style'].upper()}, not {constraints.body_style.upper()}."
        )
        penalty += 1.0
    if not drivetrain_matches(vehicle["drivetrain"], constraints.drivetrain):
        violations.append(
            f"Drivetrain is {vehicle['drivetrain']}, not {constraints.drivetrain}."
        )
        penalty += 0.25
    if (
        constraints.make
        and str(vehicle.get("make", "")).casefold() != constraints.make.casefold()
    ):
        violations.append(
            f"Make is {vehicle.get('make', 'unknown')}, not {constraints.make}."
        )
        penalty += 1.0
    return violations, penalty


def rank_closest_vehicles(
    vehicles: list[dict[str, Any]], constraints: Constraints
) -> list[
    tuple[
        dict[str, Any],
        float,
        dict[str, float | None],
        list[str],
        list[str],
    ]
]:
    scored = []
    for vehicle in vehicles:
        best_listing = select_best_listing(vehicle, constraints, allow_violations=True)
        selected = {**vehicle, "listing": best_listing} if best_listing is not None else vehicle
        score, components, missing = score_vehicle(selected, constraints)
        violations, penalty = constraint_violations(selected, constraints)
        scored.append((selected, score, components, missing, violations, penalty))
    ranked = sorted(
        scored,
        key=lambda item: (len(item[4]), item[5], bool(item[3]), -item[1]),
    )
    return [item[:5] for item in ranked]
