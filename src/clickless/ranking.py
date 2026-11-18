from __future__ import annotations

from typing import Any

from .models import Constraints


def filter_vehicles(
    vehicles: list[dict[str, Any]], constraints: Constraints
) -> list[dict[str, Any]]:
    matches = []
    for vehicle in vehicles:
        listing = vehicle["listing"]
        if constraints.max_price is not None and listing["price"] > constraints.max_price:
            continue
        if constraints.max_mileage is not None and listing["mileage"] > constraints.max_mileage:
            continue
        if constraints.min_year is not None and vehicle["year"] < constraints.min_year:
            continue
        if constraints.body_style and vehicle["body_style"] != constraints.body_style:
            continue
        if constraints.drivetrain and vehicle["drivetrain"] != constraints.drivetrain:
            continue
        matches.append(vehicle)
    return matches


def _aspect(vehicle: dict[str, Any], name: str) -> float | None:
    exact_reviews = [
        review
        for review in vehicle.get("reviews", [])
        if review.get("match_type") == "exact_year"
    ]
    qwen_reviews = [review for review in exact_reviews if review.get("aspect_method") == "qwen"]
    selected = qwen_reviews or exact_reviews
    values = [
        review["aspects"][name]
        for review in selected
        if name in review.get("aspects", {})
    ]
    if not values:
        return None
    return sum(values) / len(values)


def score_vehicle(
    vehicle: dict[str, Any], constraints: Constraints
) -> tuple[float, dict[str, float | None], list[str]]:
    listing = vehicle["listing"]
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
    listing = vehicle["listing"]
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
    if constraints.drivetrain and vehicle["drivetrain"] != constraints.drivetrain:
        violations.append(
            f"Drivetrain is {vehicle['drivetrain']}, not {constraints.drivetrain}."
        )
        penalty += 0.25
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
        score, components, missing = score_vehicle(vehicle, constraints)
        violations, penalty = constraint_violations(vehicle, constraints)
        scored.append((vehicle, score, components, missing, violations, penalty))
    ranked = sorted(
        scored,
        key=lambda item: (len(item[4]), item[5], bool(item[3]), -item[1]),
    )
    return [item[:5] for item in ranked]
