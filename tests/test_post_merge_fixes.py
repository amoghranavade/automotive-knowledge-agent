"""Checks for the post-merge fixes: AWD/4WD matching, review negation,
plain-language explanation lines, and the 60-request evaluation set."""
from clickless.data.aspects import extract_aspects
from clickless.evaluation import runner
from clickless.evaluation.scenarios import default_scenarios, unanswerable_scenarios
from clickless.graph.retrieval import _alignment_text, _opinion_phrase, _path_text
from clickless.ranking import drivetrain_matches
from clickless.service import _tidy_paths


def _polarities(text: str) -> dict[str, str]:
    return {item["aspect"]: item["polarity"] for item in extract_aspects(text, "r")}


def test_awd_request_accepts_4wd_but_not_the_reverse() -> None:
    assert drivetrain_matches("4WD", "AWD")
    assert drivetrain_matches("AWD", "AWD")
    assert not drivetrain_matches("FWD", "AWD")
    assert not drivetrain_matches("AWD", "4WD")
    assert drivetrain_matches("RWD", None)


def test_negated_praise_is_negative() -> None:
    assert _polarities("Not a cool car at all and not really reliable.")["reliability"] == "negative"
    assert _polarities("No problems at all, very dependable.")["reliability"] == "positive"
    assert _polarities("Too early to tell on reliability.")["reliability"] == "neutral"


def test_fuel_pump_is_not_fuel_economy() -> None:
    aspects = _polarities("Dealer replaced the fuel pump.")
    assert "fuel_economy" not in aspects
    assert "maintenance" in aspects


def test_explanation_lines_are_plain_language() -> None:
    assert _opinion_phrase("negative", "reliability") == "complained about the reliability"
    text = _alignment_text(["recall", "review"], "airbags", 8)
    assert text.startswith("Recalls and owner reviews both mention the airbags")
    assert "subsystem" not in text and "causation" not in text
    recall = _path_text("recall", {"campaign_number": "08V523000", "component": "ENGINE"}, None)
    assert recall == "Recall 08V523000 affecting the engine."


def test_tidy_paths_keeps_only_the_card_listing_and_drops_repeats() -> None:
    vehicle = {"id": "v1", "listing": {"source_id": "cl-2", "price": 8000, "mileage": 90000}}
    paths = [
        {"kind": "listing", "text": "Listed at $10,999 with 1 miles.", "evidence_ids": ["cl-1"]},
        {"kind": "listing", "text": "Listed at $8,000 with 90,000 miles.", "evidence_ids": ["cl-2"]},
        {"kind": "recall", "text": "Recall 1 affecting the engine.", "evidence_ids": ["r1"]},
        {"kind": "recall", "text": "Recall 1 affecting the engine.", "evidence_ids": ["r1"]},
    ]
    tidy = _tidy_paths(paths, vehicle, "memory")
    assert [path["text"] for path in tidy] == [
        "Listed at $8,000 with 90,000 miles.",
        "Recall 1 affecting the engine.",
    ]
    only_other = _tidy_paths(paths[:1], vehicle, "memory")
    assert only_other[0]["text"] == "Listed at $8,000 with 90,000 miles."


def test_default_run_has_60_requests(use_mini_catalog, tmp_path, monkeypatch) -> None:
    assert len(default_scenarios()) + len(unanswerable_scenarios()) == 60

    def fake_recommend(query, **kwargs):
        return {"match_mode": "closest_alternatives", "cohort_retention": None, "candidates": []}

    monkeypatch.setattr(runner, "recommend", fake_recommend)
    summary = runner.run("C", use_model=False, output=tmp_path / "out")
    assert summary["runs"] == 60
    assert summary["abstention_accuracy"] == 1.0
    assert summary["exact_match_rate"] == 0.0
    assert summary["false_abstention_rate"] == 1.0
