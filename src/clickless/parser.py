from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote_plus

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
    "4x4": "4WD",
    "front wheel drive": "FWD",
    "front-wheel drive": "FWD",
    "fwd": "FWD",
    "rear wheel drive": "RWD",
    "rear-wheel drive": "RWD",
    "rwd": "RWD",
}

CANONICAL_MAKES = (
    "Toyota",
    "Honda",
    "Subaru",
    "Ford",
    "Chevrolet",
    "Jeep",
    "Mazda",
    "Nissan",
    "Hyundai",
    "Kia",
    "Lexus",
    "Acura",
    "Volkswagen",
    "BMW",
    "Mercedes-Benz",
    "Audi",
    "GMC",
    "Dodge",
    "Ram",
    "Volvo",
    "Buick",
    "Cadillac",
    "Chrysler",
    "Mitsubishi",
    "Infiniti",
    "Pontiac",
    "Saturn",
)

MAKE_ALIASES: dict[str, str] = {
    **{make.casefold(): make for make in CANONICAL_MAKES},
    "chevy": "Chevrolet",
    "vw": "Volkswagen",
    "mercedes benz": "Mercedes-Benz",
    "mercedes": "Mercedes-Benz",
    "benz": "Mercedes-Benz",
}

_SORTED_MAKE_ALIASES = sorted(
    MAKE_ALIASES.items(),
    key=lambda item: (-len(item[0]), item[0] == "ram"),
)

_BASELINE_WEIGHTS: dict[str, float] = {
    "price": 0.30,
    "mileage": 0.20,
    "safety": 0.25,
    "reliability": 0.25,
}

_FEATURE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "maintenance",
        re.compile(
            r"\b(?:maintenance|maintain|affordable to maintain|repair|repairs|"
            r"brake|brakes|battery|batteries|tire|tires|upkeep|parts|service cost)\b",
            re.I,
        ),
    ),
    (
        "fuel_economy",
        re.compile(
            r"\b(?:fuel economy|fuel[-\s]?efficient|gas mileage|mpg|fuel)\b",
            re.I,
        ),
    ),
    (
        "cargo",
        re.compile(
            r"\b(?:cargo|trunk|space|spacious|room|roomy|family|storage)\b",
            re.I,
        ),
    ),
    (
        "snow",
        re.compile(r"\b(?:snow|winter|ice|icy)\b", re.I),
    ),
    (
        "safety",
        re.compile(r"\b(?:safe|safety|crash|ncap|airbags?)\b", re.I),
    ),
    (
        "reliability",
        re.compile(
            r"\b(?:reliable|reliability|dependable|long[-\s]?term|durable)\b",
            re.I,
        ),
    ),
    (
        "mileage",
        re.compile(
            r"(?<!gas\s)\bmileage\b|\b(?:low[-\s]?mileage|low miles|fewer miles|odometer)\b",
            re.I,
        ),
    ),
    (
        "price",
        re.compile(
            r"\b(?:cheap|cheapest|budget|inexpensive|price|value)\b|"
            r"\baffordable(?!\s+to\s+maintain)\b",
            re.I,
        ),
    ),
)


def _amount(value: str, has_k: str | None = None) -> int:
    number = float(value.replace(",", ""))
    return int(round(number * 1000)) if has_k else int(round(number))


def normalize_make(value: Any) -> str | None:
    """Map a vehicle make string or alias to its canonical brand name."""
    if not isinstance(value, str):
        return None
    cleaned = re.sub(r"\s+", " ", value.strip()).casefold()
    return MAKE_ALIASES.get(cleaned)


def parse(query: str) -> tuple[Constraints, list[str]]:
    text = query.lower()
    max_price = None
    max_mileage = None
    min_year = None
    body_style = None
    drivetrain = None
    make = None

    price_prefix = re.compile(
        r"(?:under|below|less than|up to|at most|no more than|around|about|"
        r"budget(?:\s+of|\s+is)?|max(?:imum)?)\s*(\$)?\s*(\d[\d,]*(?:\.\d+)?)\s*(k)?\b",
        re.I,
    )
    for match in price_prefix.finditer(text):
        has_dollar, num_str, k_str = match.group(1), match.group(2), match.group(3)
        if not has_dollar and re.match(r"\s*(?:miles?|mi|mileage|km)\b", text[match.end() :]):
            continue
        max_price = _amount(num_str, k_str)
        break

    if max_price is None:
        price_suffix = re.search(
            r"\$\s*(\d[\d,]*(?:\.\d+)?)\s*(k)?\s*(?:or less|maximum|max|budget)\b",
            text,
            re.I,
        )
        if price_suffix:
            max_price = _amount(price_suffix.group(1), price_suffix.group(2))
        else:
            for match in re.finditer(r"\$\s*(\d[\d,]*(?:\.\d+)?)\s*(k)?\b", text, re.I):
                if re.match(r"\s*(?:miles?|mi|mileage|km)\b", text[match.end() :]):
                    continue
                max_price = _amount(match.group(1), match.group(2))
                break

    mileage_patterns = (
        r"(?:under|below|less than|fewer than|max(?:imum)?|up to|at most|no more than)\s*"
        r"(\d[\d,]*(?:\.\d+)?)\s*(k)?\s*(?:miles?|mi|mileage)\b",
        r"(\d[\d,]*(?:\.\d+)?)\s*(k)?\s*(?:miles?|mi|mileage)\s*(?:or less|max(?:imum)?)\b",
    )
    for pattern in mileage_patterns:
        mileage_match = re.search(pattern, text, re.I)
        if mileage_match:
            max_mileage = _amount(mileage_match.group(1), mileage_match.group(2))
            break

    year_patterns = (
        r"\b(19\d{2}|20\d{2})\s*(?:or newer|and newer|or later|and later|\+)",
        r"(?:newer than|from|after|at least|since|min(?:imum)?(?:\s+year)?)\s*(19\d{2}|20\d{2})\b",
    )
    for pattern in year_patterns:
        year_match = re.search(pattern, text, re.I)
        if year_match:
            min_year = int(year_match.group(1))
            break

    for phrase, value in BODY_STYLES.items():
        if re.search(rf"\b{re.escape(phrase)}s?\b", text):
            body_style = value
            break

    for phrase, value in DRIVETRAINS.items():
        if re.search(rf"\b{re.escape(phrase)}\b", text):
            drivetrain = value
            break

    for alias, canonical in _SORTED_MAKE_ALIASES:
        if re.search(rf"\b{re.escape(alias)}\b", text):
            make = canonical
            break

    preferences = dict(_BASELINE_WEIGHTS)
    for feature, pattern in _FEATURE_PATTERNS:
        if pattern.search(text):
            preferences[feature] = preferences.get(feature, 0.0) + 0.20

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
            make=make,
            preferences=preferences,
        ),
        questions[:2],
    )


def build_navigation_plan(
    constraints: Constraints,
    query: str,
) -> dict[str, dict[str, Any]]:
    """Build parameterized Playwright/Selenium/Scrapy search plans across retail sites."""
    lowered = query.casefold()
    vehicle_tokens = [
        str(token)
        for token in (
            constraints.min_year,
            constraints.make,
            constraints.drivetrain,
            constraints.body_style,
        )
        if token
    ]
    vehicle_query = " ".join(vehicle_tokens) if vehicle_tokens else (query.strip() or "used car")
    parts_prefix = " ".join(
        str(token) for token in (constraints.make, constraints.body_style) if token
    ) or "car"

    replacement_terms = [
        term
        for term in ("brake", "battery", "alternator", "sensor", "rotor", "pad")
        if term in lowered
    ]
    amazon_query = (
        f"{parts_prefix} {' '.join(replacement_terms)} replacement parts"
        if replacement_terms
        else f"{parts_prefix} replacement parts"
    )

    consumable_terms = [
        term
        for term in ("battery", "tire", "tires", "oil", "fluid", "wiper")
        if term in lowered
    ]
    walmart_query = (
        f"{parts_prefix} {' '.join(consumable_terms)}"
        if consumable_terms
        else f"{parts_prefix} battery tires fluids"
    )

    electronics_terms = [
        term
        for term in ("camera", "backup camera", "dash cam", "stereo", "infotainment", "carplay")
        if term in lowered
    ]
    bestbuy_query = (
        f"{parts_prefix} {' '.join(electronics_terms)}"
        if electronics_terms
        else f"{parts_prefix} backup camera dash cam car stereo"
    )

    base_filters = {
        key: value
        for key, value in (
            ("max_price", constraints.max_price),
            ("max_mileage", constraints.max_mileage),
            ("min_year", constraints.min_year),
            ("make", constraints.make),
            ("body_style", constraints.body_style),
            ("drivetrain", constraints.drivetrain),
        )
        if value is not None
    }
    ebay_url = f"https://www.ebay.com/sch/i.html?_nkw={quote_plus(vehicle_query)}"
    if constraints.max_price is not None:
        ebay_url += f"&_udhi={constraints.max_price}"

    selector_version = "jsonld-v1 / dom-css-v2"
    return {
        "ebay": {
            "site": "ebay",
            "target": "eBay Motors vehicles & OEM parts",
            "url": ebay_url,
            "query": vehicle_query,
            "filters": {**base_filters, "category": "Cars & Trucks"},
            "selector_version": selector_version,
        },
        "amazon": {
            "site": "amazon",
            "target": "Amazon Automotive replacement parts",
            "url": f"https://www.amazon.com/s?k={quote_plus(amazon_query)}&i=automotive",
            "query": amazon_query,
            "filters": {
                **{k: v for k, v in base_filters.items() if k in {"make", "min_year", "body_style"}},
                "department": "Automotive",
            },
            "selector_version": selector_version,
        },
        "walmart": {
            "site": "walmart",
            "target": "Walmart Auto & Tires routine consumables",
            "url": f"https://www.walmart.com/search?q={quote_plus(walmart_query)}",
            "query": walmart_query,
            "filters": {
                **{k: v for k, v in base_filters.items() if k in {"make", "min_year", "body_style"}},
                "department": "Auto & Tires",
            },
            "selector_version": selector_version,
        },
        "bestbuy": {
            "site": "bestbuy",
            "target": "Best Buy Car Electronics cabin & safety upgrades",
            "url": f"https://www.bestbuy.com/site/searchpage.jsp?st={quote_plus(bestbuy_query)}",
            "query": bestbuy_query,
            "filters": {
                **{k: v for k, v in base_filters.items() if k in {"make", "min_year", "body_style"}},
                "department": "Car Electronics",
            },
            "selector_version": selector_version,
        },
    }
