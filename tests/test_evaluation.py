from __future__ import annotations

from dataclasses import replace
import csv
import json
from pathlib import Path
from typing import Any

import pytest

import clickless.retrieval as retrieval_mod
from clickless.evaluation import analyze, components, metrics, runner, scenarios


class _FakeDense:
    def encode(self, texts: list[str], normalize_embeddings: bool = True):
        import numpy as np

        arr = np.ones((len(texts), 4), dtype=np.float32)
        return arr / 2.0


def test_scenarios_50_structure_and_unanswerable() -> None:
    items = scenarios.default_scenarios()
    assert len(items) == 50
    assert [item["id"] for item in items] == [f"scenario-{i:02d}" for i in range(1, 51)]
    assert sum(item["split"] == "development" for item in items) == 15
    assert sum(item["split"] == "test" for item in items) == 35
    assert all(item["split"] == "development" for item in items[:15])
    assert all(item["split"] == "test" for item in items[15:])

    unanswerable = [item for item in items if item["answerable"] is False]
    assert len(unanswerable) == 3
    assert any("2022 or newer suv under $2,500" in item["query"].casefold() for item in unanswerable)

    expected_keys = {"max_price", "max_mileage", "min_year", "body_style", "drivetrain"}
    for item in items:
        assert isinstance(item["answerable"], bool)
        assert set(item["expected_constraints"]) == expected_keys


def test_metrics_functions_and_guards() -> None:
    assert metrics.precision_at_k([3, 2, 1, 0, 2], k=5, threshold=2.0) == pytest.approx(0.6)
    assert metrics.ndcg_at_k([3, 2, 1, 0], k=4) == pytest.approx(1.0)
    assert metrics.ndcg_at_k([0, 0, 0], k=3) == 0.0
    assert metrics.ndcg_at_k([0, 1, 3], k=3) < 1.0

    cov = metrics.citation_coverage(
        [
            {
                "pros": [{"text": "p1", "evidence_ids": ["e1"]}],
                "cautions": [{"text": "c1", "evidence_ids": []}],
            }
        ]
    )
    assert cov == pytest.approx(0.5)
    assert metrics.citation_coverage([]) == 0.0

    assert metrics.cohen_kappa([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert metrics.cohen_kappa(["a", "b"], ["a", "b"]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        metrics.cohen_kappa([1, 2], [1])

    diff, lo, hi = metrics.paired_bootstrap_ci(
        [0.9, 0.85, 0.95, 0.88, 0.92],
        [0.4, 0.45, 0.35, 0.5, 0.42],
    )
    assert diff > 0.4
    assert lo <= diff <= hi
    assert lo > 0.0
    with pytest.raises(ValueError):
        metrics.paired_bootstrap_ci([0.5], [0.5, 0.6])


def test_runner_p95_gold_constraints_abstention_and_cohort_retention(
    use_mini_catalog, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert runner._p95([]) == 0.0
    assert runner._p95([0.42]) == 0.42
    assert runner._p95([float(x) for x in range(1, 21)]) == 19.0
    assert runner._p95([float(x) for x in range(1, 101)]) == 95.0

    monkeypatch.setattr(retrieval_mod, "_DENSE_MODEL", _FakeDense())
    subset = [
        {
            "id": "scenario-01",
            "split": "development",
            "query": "Find an AWD SUV 2008 or newer under $18,000.",
            "answerable": True,
            "expected_constraints": {
                "max_price": 18_000,
                "max_mileage": None,
                "min_year": 2008,
                "body_style": "suv",
                "drivetrain": "AWD",
            },
        },
        {
            "id": "scenario-02",
            "split": "development",
            "query": "Find a vehicle 2008 or newer under $20,000.",
            "answerable": True,
            "expected_constraints": {
                "max_price": 20_000,
                "max_mileage": None,
                "min_year": 2008,
                "body_style": None,
                "drivetrain": None,
            },
        },
        {
            "id": "scenario-15",
            "split": "development",
            "query": "Find a 2022 or newer SUV under $2,500 with under 15,000 miles.",
            "answerable": False,
            "expected_constraints": {
                "max_price": 2_500,
                "max_mileage": 15_000,
                "min_year": 2022,
                "body_style": "suv",
                "drivetrain": None,
            },
        },
    ]

    out_dir = tmp_path / "eval_out"
    summary = runner.run("ABCDE", use_model=False, output=out_dir, scenarios=subset)
    assert summary["runs"] == 15
    assert summary["gold_constraint_satisfaction"] == pytest.approx(1.0)
    assert summary["abstention_accuracy"] == pytest.approx(1.0)
    assert set(summary["by_configuration"]) == {"A", "B", "C", "D", "E"}
    for cfg in "ABCDE":
        cfg_metrics = summary["by_configuration"][cfg]
        assert cfg_metrics["runs"] == 3
        assert cfg_metrics["gold_constraint_satisfaction"] == pytest.approx(1.0)
        assert cfg_metrics["abstention_accuracy"] == pytest.approx(1.0)
        assert cfg_metrics["p95_latency_seconds"] >= cfg_metrics["median_latency_seconds"]

    # Verify cohort_retention averaging and gold constraint violation detection
    def fake_mixed_recommend(query: str, **kwargs: Any) -> dict[str, Any]:
        if "2022" in query:
            return {
                "match_mode": "exact",
                "cohort_retention": 0.5,
                "candidates": [
                    {
                        "id": "overbudget-car",
                        "price": 25_000,
                        "mileage": 50_000,
                        "year": 2010,
                        "body_style": "suv",
                        "drivetrain": "AWD",
                        "pros": [{"text": "ok", "evidence_ids": ["e1"]}],
                        "cautions": [],
                    }
                ],
            }
        return {
            "match_mode": "exact",
            "cohort_retention": 0.75,
            "candidates": [
                {
                    "id": "overbudget-car",
                    "price": 25_000,
                    "mileage": 50_000,
                    "year": 2010,
                    "body_style": "suv",
                    "drivetrain": "AWD",
                    "pros": [{"text": "ok", "evidence_ids": ["e1"]}],
                    "cautions": [],
                }
            ],
        }

    monkeypatch.setattr(runner, "recommend", fake_mixed_recommend)
    bad_summary = runner.run("C", use_model=False, output=tmp_path / "bad_eval", scenarios=subset)
    assert bad_summary["constraint_satisfaction"] == pytest.approx(1.0)
    assert bad_summary["gold_constraint_satisfaction"] == pytest.approx(0.0)
    assert bad_summary["abstention_accuracy"] == pytest.approx(0.0)
    assert bad_summary["mean_cohort_retention"] == pytest.approx((0.75 + 0.75 + 0.5) / 3)


def test_analyze_paired_bootstrap_alignment_and_absa_evaluators(tmp_path: Path) -> None:
    eval_dir = tmp_path / "analyze_in"
    eval_dir.mkdir(parents=True)
    (eval_dir / "blinding_key.json").write_text(
        json.dumps({"system-1": "A", "system-2": "C"}), encoding="utf-8"
    )

    # Deliberately write A in reverse scenario order (scenario-02 then scenario-01)
    # and include scenario-03 only for C to verify exact scenario_id alignment.
    judgment_rows = []
    # A: scenario-02 (poor scores), then scenario-01 (poor scores)
    for rank, score in enumerate([0, 0, 0, 1, 3], 1):
        judgment_rows.append(
            {
                "scenario_id": "scenario-02",
                "system_id": "system-1",
                "candidate_id": f"a2-{rank}",
                "rank": str(rank),
                "judge_1_score": str(score),
                "judge_2_score": str(score),
            }
        )
    for rank, score in enumerate([0, 0, 1, 1, 2], 1):
        judgment_rows.append(
            {
                "scenario_id": "scenario-01",
                "system_id": "system-1",
                "candidate_id": f"a1-{rank}",
                "rank": str(rank),
                "judge_1_score": str(score),
                "judge_2_score": str(score),
            }
        )
    # C: scenario-01 (strong), scenario-02 (strong), scenario-03 (extra)
    for sid, scores in (
        ("scenario-01", [3, 3, 2, 1, 0]),
        ("scenario-02", [3, 3, 3, 0, 0]),
        ("scenario-03", [3, 2, 2, 0, 0]),
    ):
        for rank, score in enumerate(scores, 1):
            judgment_rows.append(
                {
                    "scenario_id": sid,
                    "system_id": "system-2",
                    "candidate_id": f"c-{sid}-{rank}",
                    "rank": str(rank),
                    "judge_1_score": str(score),
                    "judge_2_score": str(score),
                }
            )

    with (eval_dir / "human_judgments.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["scenario_id", "system_id", "candidate_id", "rank", "judge_1_score", "judge_2_score"],
        )
        writer.writeheader()
        writer.writerows(judgment_rows)

    absa_rows = [
        # Train rows with repeated tokens across all 4 classes
        {"source_id": "tr-1", "split": "train", "aspect": "reliability", "polarity": "positive", "text": "Dependable engine and great reliability.", "span": "Dependable engine", "annotator": "1"},
        {"source_id": "tr-1", "split": "train", "aspect": "reliability", "polarity": "positive", "text": "Dependable engine and great reliability.", "span": "Dependable engine", "annotator": "2"},
        {"source_id": "tr-2", "split": "train", "aspect": "reliability", "polarity": "positive", "text": "Reliable engine and smooth drivetrain.", "span": "Reliable engine", "annotator": "1"},
        {"source_id": "tr-3", "split": "train", "aspect": "maintenance", "polarity": "negative", "text": "Brakes failed early and costly repair.", "span": "Brakes failed early", "annotator": "1"},
        {"source_id": "tr-4", "split": "train", "aspect": "maintenance", "polarity": "negative", "text": "Costly brake repair and leaking gasket.", "span": "Costly brake repair", "annotator": "1"},
        {"source_id": "tr-5", "split": "train", "aspect": "maintenance", "polarity": "neutral", "text": "Scheduled oil service at 60,000 miles.", "span": "Scheduled oil service at 60,000 miles.", "annotator": "1"},
        {"source_id": "tr-6", "split": "train", "aspect": "maintenance", "polarity": "neutral", "text": "Routine oil service at 50,000 miles.", "span": "Routine oil service at 50,000 miles.", "annotator": "1"},
        {"source_id": "tr-7", "split": "train", "aspect": "fuel_economy", "polarity": "mixed", "text": "Good gas mileage, but costly brake repair.", "span": "Good gas mileage, but costly brake repair.", "annotator": "1"},
        {"source_id": "tr-8", "split": "train", "aspect": "fuel_economy", "polarity": "mixed", "text": "Great gas mileage, although costly repair.", "span": "Great gas mileage, although costly repair.", "annotator": "1"},
        # Test rows covering all 4 polarities with gold spans
        {"source_id": "te-1", "split": "test", "aspect": "reliability", "polarity": "positive", "text": "Dependable engine and smooth drive.", "span": "Dependable engine and smooth drive.", "annotator": "1"},
        {"source_id": "te-2", "split": "test", "aspect": "maintenance", "polarity": "negative", "text": "Brakes failed early and costly repair.", "span": "Brakes failed early and costly repair.", "annotator": "1"},
        {"source_id": "te-3", "split": "test", "aspect": "maintenance", "polarity": "neutral", "text": "Scheduled oil service at 60,000 miles.", "span": "Scheduled oil service at 60,000 miles.", "annotator": "1"},
        {"source_id": "te-4", "split": "test", "aspect": "fuel_economy", "polarity": "mixed", "text": "Good gas mileage, but brakes wore out quickly.", "span": "Good gas mileage, but brakes wore out quickly.", "annotator": "1"},
    ]

    with (eval_dir / "absa_annotations.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_id", "split", "aspect", "polarity", "text", "span", "annotator"],
        )
        writer.writeheader()
        writer.writerows(absa_rows)

    class _FakeAbsaQwenClient:
        model = "fake-qwen"

        def complete_json(self, system: str, user: str, max_tokens: int = 1800) -> dict:
            req = json.loads(user)
            return {
                "reviews": [
                    {
                        "source_id": rev["source_id"],
                        "assertions": [
                            {
                                "aspect": (
                                    "reliability"
                                    if "engine" in rev["text"].casefold()
                                    else "fuel_economy"
                                    if "mileage" in rev["text"].casefold()
                                    else "maintenance"
                                ),
                                "subsystem": "unknown",
                                "polarity": (
                                    "mixed"
                                    if "but" in rev["text"].casefold()
                                    else "negative"
                                    if "failed" in rev["text"].casefold()
                                    else "positive"
                                    if "dependable" in rev["text"].casefold()
                                    else "neutral"
                                ),
                                "span": rev["text"],
                            }
                        ],
                    }
                    for rev in req["reviews"]
                ]
            }

    summary = analyze.analyze(eval_dir, qwen_client=_FakeAbsaQwenClient())
    assert summary["cohen_kappa"] == pytest.approx(1.0)
    boot = summary["paired_bootstrap"]["C_vs_A"]
    assert boot["common_scenarios"] == 2
    assert boot["ndcg_at_5_difference"] > 0.3
    assert boot["precision_at_5_difference"] > 0.3
    assert boot["ndcg_at_5_ci_95"][0] > 0.0
    assert boot["precision_at_5_ci_95"][0] > 0.0

    extractor_eval = summary["absa"]["extractor_evaluation"]
    assert extractor_eval["evaluated_rows"] == 4
    extractors = extractor_eval["extractors"]
    assert set(extractors) == {"lexicon", "vader", "tfidf_logistic", "qwen"}
    assert extractors["lexicon"]["span"]["f1"] is not None
    assert extractors["lexicon"]["polarity"]["polarity_macro_f1_4class"] >= 0.0
    assert set(extractors["vader"]["polarity"]["per_class_f1"]) == {
        "positive",
        "negative",
        "neutral",
        "mixed",
    }
    assert 0.0 <= extractors["vader"]["polarity"]["polarity_macro_f1_4class"] <= 1.0
    assert extractors["tfidf_logistic"]["polarity"]["accuracy"] is not None
    assert extractors["qwen"]["span"]["f1"] == pytest.approx(1.0)
    assert extractors["qwen"]["polarity"]["polarity_macro_f1_4class"] == pytest.approx(1.0)


def test_components_slot_cases_ttl_generation_and_use_model(
    use_mini_catalog, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cases = components.slot_cases()
    assert len(cases) == 50
    assert [case["id"] for case in cases] == [f"slot-{i:02d}" for i in range(1, 51)]
    joined = " || ".join(case["query"] for case in cases)
    for expected_phrase in (
        "$18k",
        "$18.5k.",
        "below 95k miles",
        "at most $14,500",
        "2008+",
        "crossover",
        "pickup",
        "all-wheel drive",
        "4x4",
    ):
        assert expected_phrase.casefold() in joined.casefold()
    assert any(case["expected"]["min_year"] is None for case in cases)
    assert any(case["expected"]["drivetrain"] is None for case in cases)

    # Evaluate rule parser and mocked Qwen parser
    class _FakeSlotClient:
        model = "fake-qwen"

        def __init__(self) -> None:
            self.by_query = {c["query"]: c["expected"] for c in cases}

        def complete_json(self, system: str, user: str, max_tokens: int = 700) -> dict:
            req = json.loads(user)
            exp = self.by_query[req["request"]]
            return {
                "max_price": exp["max_price"],
                "max_mileage": exp["max_mileage"],
                "min_year": exp["min_year"],
                "body_style": exp["body_style"],
                "drivetrain": exp["drivetrain"],
                "preferences": {"price": 0.5, "safety": 0.5},
                "clarification_questions": [],
            }

    processed_dir = tmp_path / "processed"
    monkeypatch.setattr(
        components,
        "settings",
        replace(components.settings, processed_path=processed_dir),
    )
    out_dir = tmp_path / "comp_out"

    comp_metrics = components.run(
        out_dir,
        overwrite=True,
        use_model=True,
        client=_FakeSlotClient(),
    )
    assert (processed_dir / "clickless.ttl").exists()
    assert comp_metrics["shacl_conforms"] is True
    assert comp_metrics["slot_parsing"]["rule_parser"]["exact_match"] == pytest.approx(1.0)
    assert comp_metrics["slot_parsing"]["rule_parser"]["normalized_numeric_exact_match"] == pytest.approx(1.0)
    assert comp_metrics["slot_parsing"]["model_parser"]["exact_match"] == pytest.approx(1.0)
    assert comp_metrics["slot_parsing"]["model_parser"]["normalized_numeric_exact_match"] == pytest.approx(1.0)

    slot_file = json.loads((out_dir / "slot_parsing_metrics.json").read_text(encoding="utf-8"))
    assert "rule_parser" in slot_file
    assert "model_parser" in slot_file
    assert set(slot_file["model_parser"]["per_slot_f1"]) == set(components.SLOTS)


def test_components_isolated_slot_metrics_and_ttl_creation(
    mini_catalog, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Verify components.py TTL creation and slot_metrics logic even before sibling branches merge
    clean_catalog = json.loads(json.dumps(mini_catalog))
    clean_catalog["listings"] = [
        item for item in clean_catalog["listings"] if "|" not in item["id"]
    ]
    cat_path = tmp_path / "clean_catalog.json"
    cat_path.write_text(json.dumps(clean_catalog), encoding="utf-8")
    monkeypatch.setenv("CLICKLESS_CATALOG_PATH", str(cat_path))

    cases = components.slot_cases()
    by_query = {case["query"]: case["expected"] for case in cases}

    from clickless.models import Constraints

    def fake_parse(query: str) -> tuple[Constraints, list[str]]:
        exp = by_query[query]
        return (
            Constraints(
                max_price=exp["max_price"],
                max_mileage=exp["max_mileage"],
                min_year=exp["min_year"],
                body_style=exp["body_style"],
                drivetrain=exp["drivetrain"],
            ),
            [],
        )

    class _FakeSlotClient:
        model = "fake-qwen"

        def complete_json(self, system: str, user: str, max_tokens: int = 700) -> dict:
            req = json.loads(user)
            exp = by_query[req["request"]]
            return {
                "max_price": exp["max_price"],
                "max_mileage": exp["max_mileage"],
                "min_year": exp["min_year"],
                "body_style": exp["body_style"],
                "drivetrain": exp["drivetrain"],
                "preferences": {"price": 0.5, "safety": 0.5},
                "clarification_questions": [],
            }

    monkeypatch.setattr(components, "parse", fake_parse)
    processed_dir = tmp_path / "processed_isolated"
    monkeypatch.setattr(
        components,
        "settings",
        replace(components.settings, processed_path=processed_dir),
    )
    out_dir = tmp_path / "comp_isolated"
    comp_metrics = components.run(
        out_dir,
        overwrite=True,
        use_model=True,
        client=_FakeSlotClient(),
    )
    assert (processed_dir / "clickless.ttl").exists()
    assert comp_metrics["shacl_conforms"] is True
    assert comp_metrics["slot_parsing"]["rule_parser"]["exact_match"] == pytest.approx(1.0)
    assert comp_metrics["slot_parsing"]["model_parser"]["exact_match"] == pytest.approx(1.0)
