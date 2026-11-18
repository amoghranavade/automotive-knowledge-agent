from __future__ import annotations

from typing import Any


BUDGETS = (8_000, 12_000, 18_000, 25_000, 35_000)
BODY_STYLES = ("SUV", "sedan", "truck")
NEEDS = (
    "reliable winter driving and affordable maintenance",
    "good fuel economy and low mileage",
    "strong safety ratings and family cargo space",
    "low repair costs and dependable brakes",
    "highway comfort and long-term reliability",
)


def default_scenarios() -> list[dict[str, Any]]:
    scenarios = []
    index = 0
    for body_style in BODY_STYLES:
        for budget in BUDGETS:
            for need in NEEDS:
                if index >= 50:
                    return scenarios
                scenarios.append(
                    {
                        "id": f"scenario-{index + 1:02d}",
                        "split": "development" if index < 15 else "test",
                        "query": f"Find a {body_style} under ${budget:,} with {need}.",
                        "answerable": True,
                        "gold": {"max_price": budget, "body_style": body_style.lower()},
                    }
                )
                index += 1
    return scenarios


UNANSWERABLE = (
    "Find a truck under $1,500 with low mileage.",
    "Find a 2024 or newer sedan under $6,000.",
    "Find an RWD SUV under $3,000 with under 20,000 miles.",
    "Find a 2023 or newer truck under $9,000 with good reliability.",
    "Find an SUV under $2,000 with under 30,000 miles.",
    "Find a sedan under $1,000 with strong safety ratings.",
    "Find a 2025 or newer SUV under $12,000.",
    "Find a 4WD sedan under $4,000 with under 15,000 miles.",
    "Find a truck under $2,500 with under 25,000 miles.",
    "Find a 2022 or newer SUV under $5,000 with good cargo space.",
)


def unanswerable_scenarios() -> list[dict[str, Any]]:
    """Hard constraints no catalog vehicle can meet; the correct behavior is to abstain."""
    return [
        {
            "id": f"unanswerable-{index + 1:02d}",
            "split": "unanswerable",
            "query": query,
            "answerable": False,
            "gold": {},
        }
        for index, query in enumerate(UNANSWERABLE)
    ]
