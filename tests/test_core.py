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


def test_pre_2011_cars_get_safety_from_individual_crash_tests():
    from clickless.data.nhtsa import model_variants, safety_stars

    record = {
        "overall_stars": None,  # NHTSA: "Not Rated" before model year 2011
        "rollover_stars": 3,
        "front_driver_stars": 4,
        "front_passenger_stars": 4,
        "side_driver_stars": 5,
        "side_passenger_stars": 5,
    }
    assert safety_stars(record) == (4.2, "test_average")
    assert safety_stars({"overall_stars": 5}) == (5, "overall")
    assert "cr-v" in model_variants("cr-v ex-l")


def test_nhtsa_name_variants_cover_catalog_spellings():
    from clickless.data.nhtsa import lookup_names

    assert ("dodge", "ram 2500") in lookup_names("ram", "2500", 2007)
    assert ("dodge", "ram 1500") in lookup_names("ram", "dr", 2007)
    assert ("ford", "f-250 sd") in lookup_names("ford", "f250 super duty", 2008)
    assert ("lexus", "es350") in lookup_names("lexus", "es 350", 2007)
    assert ("ram", "2500") == lookup_names("ram", "2500", 2012)[0]
