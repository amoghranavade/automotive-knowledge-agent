from __future__ import annotations

import hashlib
import numpy as np
import pytest

import clickless.llm.pipeline as pipeline_mod
import clickless.ranking as ranking_mod
import clickless.retrieval as retrieval_mod
from clickless.models import Constraints
from clickless.ranking import (
    _aspect,
    constraint_violations,
    filter_vehicles,
    rank_vehicles,
    score_vehicle,
)
from clickless.service import recommend
from clickless.store import load_vehicles


class _FakeDenseEncoder:
    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> np.ndarray:
        dims = 16
        vectors = np.zeros((len(texts), dims), dtype=np.float32)
        for row_idx, text in enumerate(texts):
            for token in text.casefold().split():
                bucket = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % dims
                vectors[row_idx, bucket] += 1.0
            norm = np.linalg.norm(vectors[row_idx])
            if norm > 0:
                vectors[row_idx] /= norm
            else:
                vectors[row_idx, 0] = 1.0
        return vectors


class _FakeQwenAblationClient:
    model = "fake-qwen"

    def complete_json(self, system: str, user: str, max_tokens: int = 1024) -> dict:
        if "Convert a used-car request to JSON" in system:
            return {
                "max_price": 25000,
                "max_mileage": 120000,
                "min_year": 2008,
                "body_style": "suv",
                "drivetrain": "AWD",
                "make": "Subaru",
                "preferences": {
                    "price": 0.25,
                    "mileage": 0.15,
                    "safety": 0.25,
                    "reliability": 0.20,
                    "maintenance": 0.15,
                },
                "clarification_questions": [],
            }
        if "Extract review evidence" in system:
            import json

            req = json.loads(user)
            reviews_out = []
            for rev in req.get("reviews", []):
                text = rev["text"]
                snippet = text.split(".")[0].strip()
                polarity = "negative" if "wore" in text.casefold() or "cost" in text.casefold() else "positive"
                aspect = "maintenance" if "brake" in text.casefold() else "reliability"
                subsystem = "brakes" if "brake" in text.casefold() else "engine"
                assertions = (
                    [
                        {
                            "aspect": aspect,
                            "subsystem": subsystem,
                            "polarity": polarity,
                            "span": snippet,
                        }
                    ]
                    if snippet and snippet.casefold() in text.casefold()
                    else []
                )
                reviews_out.append({"source_id": rev["source_id"], "assertions": assertions})
            return {"reviews": reviews_out}
        import json

        req = json.loads(user)
        out = []
        for cand in req.get("candidates", []):
            ev = cand["evidence"][0]
            quote = ev["text"][:40]
            out.append(
                {
                    "candidate_id": cand["candidate_id"],
                    "strengths": [
                        {
                            "text": "Verified record supports this option.",
                            "supports": [{"evidence_id": ev["id"], "quote": quote}],
                        }
                    ],
                    "cautions": [],
                }
            )
        return {"candidates": out}


def test_slide7_laplace_formula_and_aspect_summary() -> None:
    assertions = (
        [{"aspect": "reliability", "subsystem": "engine", "polarity": "positive", "span": "good", "method": "qwen"}] * 6
        + [{"aspect": "reliability", "subsystem": "engine", "polarity": "negative", "span": "bad", "method": "qwen"}] * 2
        + [{"aspect": "reliability", "subsystem": "engine", "polarity": "neutral", "span": "ok", "method": "qwen"}]
        + [{"aspect": "reliability", "subsystem": "engine", "polarity": "mixed", "span": "mixed", "method": "qwen"}]
    )
    vehicle = {
        "id": "test-veh",
        "year": 2009,
        "make": "Subaru",
        "model": "Forester",
        "drivetrain": "AWD",
        "listing": {"price": 10_000, "mileage": 70_000, "data_mode": "frozen_snapshot"},
        "safety": {"ncap_stars": 5},
        "reviews": [
            {
                "source_id": "rev-exact",
                "match_type": "exact_year",
                "source_kind": "owner_review",
                "aspect_method": "qwen",
                "aspect_assertions": assertions,
                "aspects": {},
            },
            {
                "source_id": "rev-sibling",
                "match_type": "same_generation",
                "source_kind": "owner_review",
                "aspect_method": "qwen",
                "aspect_assertions": [
                    {"aspect": "cargo", "subsystem": "cabin", "polarity": "positive", "span": "big", "method": "qwen"}
                ],
                "aspects": {"cargo": 1.0},
            },
            {
                "source_id": "rev-product",
                "match_type": "product_fitment_unverified",
                "source_kind": "product_review",
                "aspect_method": "lexicon",
                "aspect_assertions": [
                    {"aspect": "maintenance", "subsystem": "brakes", "polarity": "positive", "span": "easy pads", "method": "lexicon"},
                    {"aspect": "cargo", "subsystem": "cabin", "polarity": "positive", "span": "ignored for cargo", "method": "lexicon"},
                ],
                "aspects": {"maintenance": 1.0, "cargo": 1.0},
            },
        ],
    }

    rel_score = _aspect(vehicle, "reliability")
    assert rel_score is not None
    assert rel_score == pytest.approx(15 / 22, abs=1e-3)
    assert round(rel_score, 2) == 0.68

    # Zero spans -> None (unknown, never 0.5) even when sibling/product reviews mention cargo
    assert _aspect(vehicle, "cargo") is None

    # Product review maintenance assertion counts for maintenance aspect
    maint_score = _aspect(vehicle, "maintenance")
    assert maint_score == pytest.approx(0.75, abs=1e-3)

    summary = ranking_mod.aspect_summary(vehicle)
    assert summary["reliability"]["positive"] == 6
    assert summary["reliability"]["negative"] == 2
    assert summary["reliability"]["neutral"] == 1
    assert summary["reliability"]["mixed"] == 1
    assert summary["reliability"]["total"] == 10
    assert summary["reliability"]["score"] == pytest.approx(15 / 22, abs=1e-3)
    assert summary["cargo"]["total"] == 0
    assert summary["cargo"]["score"] is None


def test_listing_first_filtering_and_make_constraint(use_mini_catalog) -> None:
    vehicles = load_vehicles(live_nhtsa=False)
    forester_2009 = next(v for v in vehicles if v["id"] == "subaru-forester-2009-awd")
    # Has 2 listings: cheapest ($9,200 / 135k mi) and compliant ($11,400 / 72k mi)
    assert len(forester_2009["listings"]) == 2

    constraints = Constraints(
        max_price=12_000,
        max_mileage=80_000,
        body_style="suv",
        drivetrain="AWD",
        make="Subaru",
        preferences={"price": 0.3, "mileage": 0.2, "safety": 0.25, "reliability": 0.25},
    )
    filtered = filter_vehicles(vehicles, constraints)
    assert [v["id"] for v in filtered] == ["subaru-forester-2009-awd"]
    ranked = rank_vehicles(vehicles, constraints)
    assert ranked[0][0]["id"] == "subaru-forester-2009-awd"
    best_listing = ranking_mod.select_best_listing(forester_2009, constraints)
    assert best_listing["price"] == 11_400
    assert best_listing["mileage"] == 72_000

    accord = next(v for v in vehicles if v["id"] == "honda-accord-2008-fwd")
    violations, penalty = constraint_violations(accord, constraints)
    assert any("subaru" in msg.casefold() for msg in violations)
    assert penalty > 0

    # Unrated Accord gets missing=['safety']
    _, components, missing = score_vehicle(accord, Constraints(preferences={"safety": 0.5, "price": 0.5}))
    assert components["safety"] is None
    assert "safety" in missing


def test_cohort_retention_and_unrated_tier(
    use_mini_catalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(retrieval_mod, "_DENSE_MODEL", _FakeDenseEncoder())
    res = recommend(
        "Find a vehicle 2008 or newer under $20,000",
        live_nhtsa=False,
        use_model=False,
        configuration="C",
        candidate_limit=5,
    )
    assert res["feasible_count"] == 4
    assert res["comparable_cohort_count"] == 3
    assert res["cohort_retention"] == pytest.approx(0.75, abs=1e-4)
    assert "navigation_plan" in res

    by_id = {c["id"]: c for c in res["candidates"]}
    assert by_id["honda-accord-2008-fwd"]["tier"] == "requires_additional_evidence"
    assert by_id["subaru-forester-2009-awd"]["tier"] == "complete"
    assert "aspect_summary" in by_id["subaru-forester-2009-awd"]


def test_ablation_invariants_a_through_e(
    use_mini_catalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(retrieval_mod, "_DENSE_MODEL", _FakeDenseEncoder())
    pipeline_mod._ASPECT_CACHE.clear()
    query = "Find a Subaru AWD SUV 2008 or newer under $25,000 with reliable brakes"

    results_no_model = {
        cfg: recommend(
            query,
            live_nhtsa=False,
            use_model=False,
            configuration=cfg,
            candidate_limit=5,
        )
        for cfg in "ABCDE"
    }

    # B, C, D must have identical candidate ordering and scores
    for cfg in ("C", "D"):
        assert [c["id"] for c in results_no_model[cfg]["candidates"]] == [
            c["id"] for c in results_no_model["B"]["candidates"]
        ]
        assert [c["score"] for c in results_no_model[cfg]["candidates"]] == [
            c["score"] for c in results_no_model["B"]["candidates"]
        ]
        assert [c["score_components"] for c in results_no_model[cfg]["candidates"]] == [
            c["score_components"] for c in results_no_model["B"]["candidates"]
        ]

    # A uses bm25 + vader and no graph paths
    assert results_no_model["A"]["model_stages"]["review_extraction"] == "vader"
    assert all(c["retrieval_mode"] == "bm25" and c["graph_paths"] == [] for c in results_no_model["A"]["candidates"])

    # B has no graph paths
    assert all(c["graph_paths"] == [] for c in results_no_model["B"]["candidates"])

    # D keeps generation_lineage (non-duplicated) for 2010 Forester and strips AspectAssertion paths
    d_2010 = next(
        c for c in results_no_model["D"]["candidates"] if c["id"] == "subaru-forester-2010-awd"
    )
    d_lineage = [p for p in d_2010["graph_paths"] if p["kind"] == "generation_lineage"]
    assert len(d_lineage) >= 1
    assert len(d_lineage) == len({tuple(p["evidence_ids"]) for p in d_lineage})
    d_reviews = [p for p in d_2010["graph_paths"] if p["kind"] == "review"]
    assert d_reviews
    assert all("HAS_ASSERTION" not in p["relations"] for p in d_reviews)

    # E removes sentiment weights from ranking
    e_res = results_no_model["E"]
    for sentiment_key in ("reliability", "maintenance", "fuel_economy", "cargo"):
        assert sentiment_key not in e_res["parsed"]["preferences"]

    # Also verify B=C=D parity when Qwen client is enabled (mocked)
    fake_client = _FakeQwenAblationClient()
    monkeypatch.setattr("clickless.service.QwenClient.from_env", lambda: fake_client)
    pipeline_mod._ASPECT_CACHE.clear()
    qwen_results = {
        cfg: recommend(
            query,
            live_nhtsa=False,
            use_model=True,
            configuration=cfg,
            candidate_limit=5,
        )
        for cfg in ("B", "C", "D")
    }
    for cfg in ("C", "D"):
        assert [c["id"] for c in qwen_results[cfg]["candidates"]] == [
            c["id"] for c in qwen_results["B"]["candidates"]
        ]
        assert [c["score"] for c in qwen_results[cfg]["candidates"]] == [
            c["score"] for c in qwen_results["B"]["candidates"]
        ]
