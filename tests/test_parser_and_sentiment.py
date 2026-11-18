from __future__ import annotations

import pytest

import clickless.parser as parser_mod
from clickless.llm.pipeline import parse_request_with_model
from clickless.parser import parse
from clickless.sentiment import (
    POLARITY_SCORE,
    TfidfLogisticSentiment,
    apply_vader_aspects,
    vader_labels,
)
from clickless.service import parse_request


class _FakeParserClient:
    model = "fake-qwen"

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def complete_json(self, system: str, user: str, max_tokens: int = 700) -> dict:
        return self.payload


def test_parse_trailing_period_k_and_small_dollar_amounts() -> None:
    constraints, questions = parse("Looking for a Toyota crossover 2008+ under $18.5k.")
    assert constraints.max_price == 18_500
    assert constraints.min_year == 2008
    assert constraints.body_style == "suv"
    assert constraints.make == "Toyota"
    assert questions == []

    cheap, _ = parse("VW sedan under $500")
    assert cheap.max_price == 500
    assert cheap.make == "Volkswagen"
    assert cheap.body_style == "sedan"


def test_parse_aliases_and_bounds() -> None:
    constraints, _ = parse(
        "Chevy pickup 4x4 at most $14,500 below 95k miles at least 2009"
    )
    assert constraints.make == "Chevrolet"
    assert constraints.body_style == "truck"
    assert constraints.drivetrain == "4WD"
    assert constraints.max_price == 14_500
    assert constraints.max_mileage == 95_000
    assert constraints.min_year == 2009

    or_less, _ = parse("All-wheel drive SUV after 2008, $18,000 or less")
    assert or_less.drivetrain == "AWD"
    assert or_less.body_style == "suv"
    assert or_less.min_year == 2008
    assert or_less.max_price == 18_000
    assert or_less.max_mileage is None


def test_parse_partial_query_clarifications_and_preferences() -> None:
    baseline, questions = parse("Find a Subaru")
    assert baseline.make == "Subaru"
    assert baseline.max_price is None
    assert baseline.body_style is None
    assert len(questions) == 2
    assert set(baseline.preferences) == {"price", "mileage", "safety", "reliability"}
    assert sum(baseline.preferences.values()) == pytest.approx(1.0, abs=1e-2)

    focused, _ = parse(
        "Find an AWD SUV under $16,000 for winter snow, good gas mileage, "
        "family cargo space, and affordable brake maintenance."
    )
    for feature in ("snow", "fuel_economy", "cargo", "maintenance"):
        assert focused.preferences.get(feature, 0.0) > 0.0
    assert sum(focused.preferences.values()) == pytest.approx(1.0, abs=1e-2)


def test_build_navigation_plan_and_parse_request() -> None:
    constraints, _ = parse("Find a Subaru Forester AWD SUV under $15,000")
    plan = parser_mod.build_navigation_plan(constraints, "Subaru Forester brake pads")
    assert set(plan) == {"ebay", "amazon", "walmart", "bestbuy"}
    for site_key, entry in plan.items():
        assert entry["site"] == site_key
        assert entry["target"]
        assert entry["url"].startswith("https://")
        assert entry["query"]
        assert isinstance(entry["filters"], dict)
        assert entry["selector_version"] == "jsonld-v1 / dom-css-v2"

    response = parse_request("Find an SUV under $15,000")
    assert "navigation_plan" in response
    assert set(response["navigation_plan"]) == {"ebay", "amazon", "walmart", "bestbuy"}


def test_parse_request_with_model_normalizes_make_and_filters_optional_weights() -> None:
    fallback, fallback_q = parse("chevy suv under $15,000")
    client = _FakeParserClient(
        {
            "max_price": 15000,
            "max_mileage": 90000,
            "min_year": 2010,
            "body_style": "SUV",
            "drivetrain": "awd",
            "make": "chevy",
            "preferences": {
                "price": 0.4,
                "safety": 0.4,
                "snow": 0.05,
                "cargo": 0.25,
            },
            "clarification_questions": [],
        }
    )
    parsed, questions = parse_request_with_model(
        client, "chevy suv under $15,000", fallback, fallback_q
    )
    assert parsed.make == "Chevrolet"
    assert parsed.body_style == "suv"
    assert parsed.drivetrain == "AWD"
    assert "cargo" in parsed.preferences
    assert "snow" not in parsed.preferences
    assert questions == []


def test_vader_domain_tuning_and_four_class_mixed_polarity() -> None:
    assert POLARITY_SCORE["mixed"] == 0.5
    labels = vader_labels(
        [
            "Good gas mileage and dependable engine.",
            "High mileage with leaking gaskets and worn brakes.",
            "The engine is reliable and smooth, but the brakes wore out quickly.",
            "The vehicle has four doors and standard wheels.",
        ]
    )
    assert labels[0] == "positive"
    assert labels[1] == "negative"
    assert labels[2] == "mixed"
    assert labels[3] == "neutral"


def test_apply_vader_aspects_updates_assertions_and_reviews() -> None:
    vehicles = [
        {
            "id": "v1",
            "reviews": [
                {
                    "source_id": "rev-1",
                    "text": "Great gas mileage, though brakes wore early.",
                    "aspect_assertions": [
                        {
                            "aspect": "fuel_economy",
                            "subsystem": "fuel_system",
                            "polarity": "neutral",
                            "span": "Great gas mileage",
                            "method": "lexicon",
                        },
                        {
                            "aspect": "maintenance",
                            "subsystem": "brakes",
                            "polarity": "neutral",
                            "span": "brakes wore early and costly repair",
                            "method": "lexicon",
                        },
                    ],
                    "aspects": {},
                    "aspect_method": "lexicon",
                }
            ],
        }
    ]
    updated = apply_vader_aspects(vehicles)
    review = updated[0]["reviews"][0]
    assert review["aspect_method"] == "vader"
    assert review["aspect_assertions"][0]["polarity"] == "positive"
    assert review["aspect_assertions"][0]["method"] == "vader"
    assert review["aspect_assertions"][1]["polarity"] == "negative"
    assert review["aspect_assertions"][1]["method"] == "vader"
    assert review["aspects"]["fuel_economy"] == 1.0
    assert review["aspects"]["maintenance"] == 0.0


def test_tfidf_logistic_sentiment_fit_predict_and_guards() -> None:
    model = TfidfLogisticSentiment()
    with pytest.raises(RuntimeError):
        model.predict(["reliable car"])
    with pytest.raises(ValueError):
        model.fit([], [])

    texts = [
        "reliable engine good mileage",
        "dependable suv great mileage",
        "costly repair worn brakes",
        "terrible transmission failure costly",
        "standard four door sedan",
        "standard interior four wheels",
        "reliable engine but costly brakes",
        "good mpg however terrible noise",
    ] * 2
    labels = [
        "positive",
        "positive",
        "negative",
        "negative",
        "neutral",
        "neutral",
        "mixed",
        "mixed",
    ] * 2
    fitted = model.fit(texts, labels)
    preds = fitted.predict(["dependable engine", "costly brake failure"])
    assert preds == ["positive", "negative"]
