from __future__ import annotations

import re

from .models import Constraints


BODY_STYLES = {
    "suv": "suv",
    "crossover": "suv",
    "sedan": "sedan",
    "truck": "truck",
    "pickup": "truck",
}

DRIVETRAINS = {
    "all wheel drive": "AWD",
    "all-wheel drive": "AWD",
    "awd": "AWD",
    "four wheel drive": "4WD",
    "four-wheel drive": "4WD",
    "4wd": "4WD",
    "front wheel drive": "FWD",
    "front-wheel drive": "FWD",
    "fwd": "FWD",
    "rear wheel drive": "RWD",
    "rear-wheel drive": "RWD",
    "rwd": "RWD",
}


def _amount(value: str) -> int:
    number = float(value.replace(",", ""))
    return int(number * 1000) if number < 1000 else int(number)


def parse(query: str) -> tuple[Constraints, list[str]]:
    text = query.lower()
    max_price = None
    max_mileage = None
    min_year = None
    body_style = None
    drivetrain = None

    not_miles = r"(?![\d,.])(?!\s*k?\s*(?:miles|mi\b|mileage|odometer))"
    price_patterns = [
        r"(?:under|below|less than|no more than|at most|max(?:imum)?|budget(?: of| is)?|up to|around|about)\s*\$?(\d[\d,.]*)" + not_miles,
        r"\$(\d[\d,.]*)\s*k?\s*(?:or less|maximum|max)",
        r"\$(\d[\d,.]*)" + not_miles,
    ]
    for pattern in price_patterns:
        match = re.search(pattern, text)
        if match:
            max_price = _amount(match.group(1).rstrip(".,"))
            break

    mileage_match = re.search(
        r"(?:under|below|less than|fewer than|no more than|at most|max(?:imum)?|up to)\s*([\d,.]+)\s*k?\s*(?:miles|mi|mileage)",
        text,
    )
    if mileage_match:
        max_mileage = _amount(mileage_match.group(1))

    year_match = re.search(r"(20\d{2})\s*(?:or newer|and newer|\+)", text)
    if not year_match:
        year_match = re.search(r"(?:newer than|from)\s*(20\d{2})", text)
    if year_match:
        min_year = int(year_match.group(1))

    for phrase, value in BODY_STYLES.items():
        if re.search(rf"\b{re.escape(phrase)}s?\b", text):
            body_style = value
            break

    for phrase, value in DRIVETRAINS.items():
        if re.search(rf"\b{re.escape(phrase)}\b", text):
            drivetrain = value
            break

    preferences = {
        "price": 0.20,
        "mileage": 0.15,
        "safety": 0.25,
        "reliability": 0.25,
        "maintenance": 0.15,
    }
    boosts = {
        "safe": "safety",
        "safety": "safety",
        "reliable": "reliability",
        "reliability": "reliability",
        "maintenance": "maintenance",
        "affordable to maintain": "maintenance",
        "fuel economy": "fuel_economy",
        "fuel-efficient": "fuel_economy",
        "gas mileage": "fuel_economy",
        "cargo": "cargo",
        "low mileage": "mileage",
        "snow": "snow",
        "winter": "snow",
    }
    for phrase, feature in boosts.items():
        if phrase in text:
            preferences[feature] = preferences.get(feature, 0.0) + 0.15

    total = sum(preferences.values())
    preferences = {key: round(value / total, 4) for key, value in preferences.items()}

    questions: list[str] = []
    if max_price is None:
        questions.append("What is your maximum budget?")
    if body_style is None:
        questions.append("Which body style do you prefer: SUV, sedan, or truck?")

    return (
        Constraints(
            max_price=max_price,
            max_mileage=max_mileage,
            min_year=min_year,
            body_style=body_style,
            drivetrain=drivetrain,
            preferences=preferences,
        ),
        questions[:2],
    )
