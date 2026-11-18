from __future__ import annotations

from typing import Any


def _constraints(
    *,
    max_price: int | None = None,
    max_mileage: int | None = None,
    min_year: int | None = None,
    body_style: str | None = None,
    drivetrain: str | None = None,
) -> dict[str, Any]:
    return {
        "max_price": max_price,
        "max_mileage": max_mileage,
        "min_year": min_year,
        "body_style": body_style,
        "drivetrain": drivetrain,
    }


_SCENARIO_SPECS: tuple[tuple[str, str, bool, dict[str, Any]], ...] = (
    # Development split: scenarios 01-15
    (
        "budget_mileage",
        "Find an SUV under $12,000 with under 100,000 miles and good fuel economy.",
        True,
        _constraints(max_price=12_000, max_mileage=100_000, body_style="suv"),
    ),
    (
        "winter_awd",
        "Find an AWD SUV 2008 or newer under $18,000 with reliable winter driving and affordable maintenance.",
        True,
        _constraints(max_price=18_000, min_year=2008, body_style="suv", drivetrain="AWD"),
    ),
    (
        "family_cargo",
        "Find a sedan under $10,000 with strong safety ratings and family cargo space.",
        True,
        _constraints(max_price=10_000, body_style="sedan"),
    ),
    (
        "aftermarket_maintenance",
        "Find a FWD sedan 2007 or newer under $14,000 with under 90,000 miles, dependable brakes, and low battery replacement costs.",
        True,
        _constraints(max_price=14_000, max_mileage=90_000, min_year=2007, body_style="sedan", drivetrain="FWD"),
    ),
    (
        "budget_mileage",
        "Find a truck under $20,000 with under 110,000 miles and low repair costs.",
        True,
        _constraints(max_price=20_000, max_mileage=110_000, body_style="truck"),
    ),
    (
        "winter_awd",
        "Find a 4WD truck 2008 or newer under $25,000 for snow driving and affordable maintenance.",
        True,
        _constraints(max_price=25_000, min_year=2008, body_style="truck", drivetrain="4WD"),
    ),
    (
        "model_year",
        "Find an SUV 2009 or newer under $16,000 with strong safety ratings and trunk space.",
        True,
        _constraints(max_price=16_000, min_year=2009, body_style="suv"),
    ),
    (
        "conflicting_priorities",
        "Find an AWD SUV 2008 or newer under $18,000 with under 95,000 miles, great gas mileage, and generous family cargo room.",
        True,
        _constraints(max_price=18_000, max_mileage=95_000, min_year=2008, body_style="suv", drivetrain="AWD"),
    ),
    (
        "sparse_evidence",
        "Find a FWD sedan 2008 or newer under $16,000 with under 100,000 miles and highway comfort.",
        True,
        _constraints(max_price=16_000, max_mileage=100_000, min_year=2008, body_style="sedan", drivetrain="FWD"),
    ),
    (
        "aftermarket_maintenance",
        "Find an SUV below $15,000 with under 120,000 miles, reliable brake pads, and affordable upkeep.",
        True,
        _constraints(max_price=15_000, max_mileage=120_000, body_style="suv"),
    ),
    (
        "family_cargo",
        "Find a crossover 2008 or newer with a budget of $18,000, under 95,000 miles, and high safety ratings.",
        True,
        _constraints(max_price=18_000, max_mileage=95_000, min_year=2008, body_style="suv"),
    ),
    (
        "budget_mileage",
        "Find a sedan below $8,500 with under 120,000 miles and good gas mileage.",
        True,
        _constraints(max_price=8_500, max_mileage=120_000, body_style="sedan"),
    ),
    (
        "winter_awd",
        "Find an all-wheel drive SUV 2008 or newer under $17,000 for winter snow and low repair costs.",
        True,
        _constraints(max_price=17_000, min_year=2008, body_style="suv", drivetrain="AWD"),
    ),
    (
        "conflicting_priorities",
        "Find a 4WD pickup under $15,000 with under 80,000 miles and good fuel economy.",
        True,
        _constraints(max_price=15_000, max_mileage=80_000, body_style="truck", drivetrain="4WD"),
    ),
    (
        "unanswerable",
        "Find a 2022 or newer SUV under $2,500 with under 15,000 miles and strong safety ratings.",
        False,
        _constraints(max_price=2_500, max_mileage=15_000, min_year=2022, body_style="suv"),
    ),
    # Test split: scenarios 16-50
    (
        "budget_mileage",
        "Find a sedan under $12,000 with under 85,000 miles and long-term reliability.",
        True,
        _constraints(max_price=12_000, max_mileage=85_000, body_style="sedan"),
    ),
    (
        "budget_mileage",
        "Find an SUV under $18,000 with under 100,000 miles and strong safety ratings.",
        True,
        _constraints(max_price=18_000, max_mileage=100_000, body_style="suv"),
    ),
    (
        "budget_mileage",
        "Find a truck under $30,000 with under 90,000 miles and dependable brakes.",
        True,
        _constraints(max_price=30_000, max_mileage=90_000, body_style="truck"),
    ),
    (
        "model_year",
        "Find a FWD sedan 2009 or newer under $15,000 with good fuel economy.",
        True,
        _constraints(max_price=15_000, min_year=2009, body_style="sedan", drivetrain="FWD"),
    ),
    (
        "model_year",
        "Find an AWD SUV 2007 or newer under $14,000 with under 115,000 miles and reliable maintenance.",
        True,
        _constraints(max_price=14_000, max_mileage=115_000, min_year=2007, body_style="suv", drivetrain="AWD"),
    ),
    (
        "model_year",
        "Find a 4WD truck 2009 or newer under $28,000 with low repair costs.",
        True,
        _constraints(max_price=28_000, min_year=2009, body_style="truck", drivetrain="4WD"),
    ),
    (
        "winter_awd",
        "Find an AWD SUV under $22,000 with under 90,000 miles for winter ice and snow.",
        True,
        _constraints(max_price=22_000, max_mileage=90_000, body_style="suv", drivetrain="AWD"),
    ),
    (
        "winter_awd",
        "Find a 4x4 truck 2008 or newer under $24,000 with under 110,000 miles for winter driving.",
        True,
        _constraints(max_price=24_000, max_mileage=110_000, min_year=2008, body_style="truck", drivetrain="4WD"),
    ),
    (
        "winter_awd",
        "Find an all-wheel drive crossover 2009 or newer under $20,000 for snow safety and reliability.",
        True,
        _constraints(max_price=20_000, min_year=2009, body_style="suv", drivetrain="AWD"),
    ),
    (
        "family_cargo",
        "Find an SUV 2008 or newer under $20,000 with family cargo space and 5-star safety priority.",
        True,
        _constraints(max_price=20_000, min_year=2008, body_style="suv"),
    ),
    (
        "family_cargo",
        "Find a sedan 2008 or newer under $16,000 with under 95,000 miles, roomy trunk space, and strong safety ratings.",
        True,
        _constraints(max_price=16_000, max_mileage=95_000, min_year=2008, body_style="sedan"),
    ),
    (
        "family_cargo",
        "Find a crossover under $25,000 with under 80,000 miles, family room, and high reliability.",
        True,
        _constraints(max_price=25_000, max_mileage=80_000, body_style="suv"),
    ),
    (
        "aftermarket_maintenance",
        "Find a FWD sedan 2008 or newer under $13,000 with under 100,000 miles, cheap brake repairs, and affordable battery replacement.",
        True,
        _constraints(max_price=13_000, max_mileage=100_000, min_year=2008, body_style="sedan", drivetrain="FWD"),
    ),
    (
        "aftermarket_maintenance",
        "Find an SUV under $17,000 with dependable brakes, low tire and battery upkeep, and affordable parts.",
        True,
        _constraints(max_price=17_000, body_style="suv"),
    ),
    (
        "aftermarket_maintenance",
        "Find a pickup 2007 or newer under $22,000 with under 120,000 miles and low brake and suspension repair costs.",
        True,
        _constraints(max_price=22_000, max_mileage=120_000, min_year=2007, body_style="truck"),
    ),
    (
        "aftermarket_maintenance",
        "Find a sedan under $15,000 with affordable maintenance, strong alternator and battery reliability, and good mpg.",
        True,
        _constraints(max_price=15_000, body_style="sedan"),
    ),
    (
        "conflicting_priorities",
        "Find an AWD SUV 2008 or newer under $14,000 with under 110,000 miles, strong safety, and low fuel costs.",
        True,
        _constraints(max_price=14_000, max_mileage=110_000, min_year=2008, body_style="suv", drivetrain="AWD"),
    ),
    (
        "conflicting_priorities",
        "Find a 4WD SUV under $16,000 with under 90,000 miles and great fuel economy.",
        True,
        _constraints(max_price=16_000, max_mileage=90_000, body_style="suv", drivetrain="4WD"),
    ),
    (
        "conflicting_priorities",
        "Find a truck 2008 or newer under $22,000 with under 95,000 miles, high mpg, and strong safety ratings.",
        True,
        _constraints(max_price=22_000, max_mileage=95_000, min_year=2008, body_style="truck"),
    ),
    (
        "conflicting_priorities",
        "Find a sedan 2008 or newer under $12,000 with under 85,000 miles, spacious cargo room, and winter traction.",
        True,
        _constraints(max_price=12_000, max_mileage=85_000, min_year=2008, body_style="sedan"),
    ),
    (
        "sparse_evidence",
        "Find a RWD truck 2008 or newer under $22,000 with under 135,000 miles and affordable maintenance.",
        True,
        _constraints(max_price=22_000, max_mileage=135_000, min_year=2008, body_style="truck", drivetrain="RWD"),
    ),
    (
        "sparse_evidence",
        "Find a FWD sedan 2009 or newer under $15,000 with under 110,000 miles and strong safety ratings.",
        True,
        _constraints(max_price=15_000, max_mileage=110_000, min_year=2009, body_style="sedan", drivetrain="FWD"),
    ),
    (
        "sparse_evidence",
        "Find a 4WD SUV 2007 or newer under $26,000 with dependable brakes and cargo space.",
        True,
        _constraints(max_price=26_000, min_year=2007, body_style="suv", drivetrain="4WD"),
    ),
    (
        "sparse_evidence",
        "Find a 4WD truck 2009 or newer under $28,000 with under 120,000 miles and long-term reliability.",
        True,
        _constraints(max_price=28_000, max_mileage=120_000, min_year=2009, body_style="truck", drivetrain="4WD"),
    ),
    (
        "budget_mileage",
        "Find a FWD sedan under $9,500 with under 105,000 miles and good fuel economy.",
        True,
        _constraints(max_price=9_500, max_mileage=105_000, body_style="sedan", drivetrain="FWD"),
    ),
    (
        "budget_mileage",
        "Find an SUV at most $24,000 with under 75,000 miles and strong safety ratings.",
        True,
        _constraints(max_price=24_000, max_mileage=75_000, body_style="suv"),
    ),
    (
        "model_year",
        "Find a sedan 2009 or newer under $16,000 with under 100,000 miles and reliable maintenance.",
        True,
        _constraints(max_price=16_000, max_mileage=100_000, min_year=2009, body_style="sedan"),
    ),
    (
        "model_year",
        "Find an SUV 2009 or newer under $23,000 with under 110,000 miles and family cargo space.",
        True,
        _constraints(max_price=23_000, max_mileage=110_000, min_year=2009, body_style="suv"),
    ),
    (
        "winter_awd",
        "Find an AWD SUV under $32,000 with under 65,000 miles for winter driving and safety.",
        True,
        _constraints(max_price=32_000, max_mileage=65_000, body_style="suv", drivetrain="AWD"),
    ),
    (
        "family_cargo",
        "Find a pickup 2008 or newer under $35,000 with under 95,000 miles, family cabin room, and safety ratings.",
        True,
        _constraints(max_price=35_000, max_mileage=95_000, min_year=2008, body_style="truck"),
    ),
    (
        "aftermarket_maintenance",
        "Find a FWD sedan under $11,500 with under 115,000 miles, low brake repair costs, and good gas mileage.",
        True,
        _constraints(max_price=11_500, max_mileage=115_000, body_style="sedan", drivetrain="FWD"),
    ),
    (
        "budget_mileage",
        "Find an SUV under $35,000 with under 60,000 miles, high reliability, and comfortable ride.",
        True,
        _constraints(max_price=35_000, max_mileage=60_000, body_style="suv"),
    ),
    (
        "sparse_evidence",
        "Find a 4WD truck 2008 or newer under $34,000 with under 100,000 miles and strong safety ratings.",
        True,
        _constraints(max_price=34_000, max_mileage=100_000, min_year=2008, body_style="truck", drivetrain="4WD"),
    ),
    (
        "unanswerable",
        "Find a 2024 or newer 4WD truck under $3,000 with under 10,000 miles and affordable maintenance.",
        False,
        _constraints(max_price=3_000, max_mileage=10_000, min_year=2024, body_style="truck", drivetrain="4WD"),
    ),
    (
        "unanswerable",
        "Find a 2023 or newer AWD sedan under $2,000 with under 20,000 miles and good fuel economy.",
        False,
        _constraints(max_price=2_000, max_mileage=20_000, min_year=2023, body_style="sedan", drivetrain="AWD"),
    ),
)


def default_scenarios() -> list[dict[str, Any]]:
    """Return the deterministic 50-scenario evaluation benchmark (15 dev, 35 test)."""
    return [
        {
            "id": f"scenario-{index + 1:02d}",
            "split": "development" if index < 15 else "test",
            "category": category,
            "query": query,
            "answerable": answerable,
            "expected_constraints": dict(expected),
        }
        for index, (category, query, answerable, expected) in enumerate(_SCENARIO_SPECS)
    ]


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
            "expected_constraints": {},
            "gold": {},
        }
        for index, query in enumerate(UNANSWERABLE)
    ]
