import pytest

from clickless.models import Constraints
from clickless.parser import parse
from clickless.ranking import _aspect
from clickless.verifier import VerificationError, _verify_numeric_claims


@pytest.mark.parametrize(
    "query, price, mileage",
    [
        ("Find a reliable SUV under $18,000 with good safety ratings", 18000, None),
        ("Find an SUV with under 100k miles under $15,000", 15000, 100000),
        ("I want a sedan, budget is 12000, AWD", 12000, None),
        ("Truck under 15k with less than 90,000 miles, 2008 or newer", 15000, 90000),
        ("SUV, $14,500 max, under 120,000 miles", 14500, 120000),
        ("sedan, no more than $9,000, at most 80,000 miles", 9000, 80000),
    ],
)
def test_price_and_mileage_are_not_confused(query, price, mileage):
    constraints, _ = parse(query)
    assert constraints.max_price == price
    assert constraints.max_mileage == mileage


def test_aspect_score_matches_slide_formula():
    polarities = ["positive"] * 6 + ["negative"] * 2 + ["neutral", "mixed"]
    vehicle = {
        "reviews": [
            {
                "match_type": "exact_year",
                "aspect_assertions": [{"aspect": "reliability", "polarity": p} for p in polarities],
            }
        ]
    }
    assert _aspect(vehicle, "reliability") == pytest.approx(0.6818, abs=1e-4)
    assert _aspect(vehicle, "cargo") is None  # no spans -> unknown, not 0.5


def test_indirect_reviews_do_not_score():
    vehicle = {
        "reviews": [
            {
                "match_type": "same_generation",
                "aspect_assertions": [{"aspect": "reliability", "polarity": "positive"}],
            }
        ]
    }
    assert _aspect(vehicle, "reliability") is None


def test_verifier_rejects_invented_price():
    evidence = [{"type": "listing", "facts": {"price": 14500}, "text": "Listed at $14,500."}]
    with pytest.raises(VerificationError):
        _verify_numeric_claims("Great value at $12,000.", evidence)
