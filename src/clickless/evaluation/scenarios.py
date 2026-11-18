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
                    }
                )
                index += 1
    return scenarios
