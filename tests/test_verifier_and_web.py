from __future__ import annotations

import copy
import importlib.util
import io
import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

import pytest
import requests

import clickless.retrieval as retrieval_mod
from clickless.data.nhtsa import fetch_recalls
from clickless.llm.pipeline import apply_grounded_summaries
from clickless.llm.qwen import QwenClient
from clickless.retail.base import RetailResult
from clickless.sources.auto_dev import AutoDevClient
from clickless.sources.serpapi import SerpApiClient
from clickless.verifier import VerificationError, verify_candidate
from clickless.web import create_app


class _FakeDense:
    def encode(self, texts: list[str], normalize_embeddings: bool = True):
        import numpy as np

        arr = np.ones((len(texts), 4), dtype=np.float32)
        return arr / 2.0


def _valid_candidate() -> dict[str, Any]:
    return {
        "id": "subaru-forester-2009-awd",
        "year": 2009,
        "make": "Subaru",
        "model": "Forester",
        "trim": "2.5X",
        "body_style": "suv",
        "drivetrain": "AWD",
        "price": 11400,
        "mileage": 72000,
        "safety_rating": 5,
        "recall_count": 1,
        "score_components": {"safety": 1.0},
        "pros": [
            {
                "text": "Listed at $11,400 with 72,000 miles.",
                "evidence_ids": ["list-1"],
            },
            {
                "text": "Brake wear noted around 40,000 miles.",
                "evidence_ids": ["rev-1"],
            },
        ],
        "cautions": [
            {
                "text": "Ceramic brake pads listed around $65.",
                "evidence_ids": ["part-1"],
            }
        ],
        "scoring_evidence_ids": ["list-1", "safe-1", "rev-1"],
        "evidence": [
            {
                "id": "list-1",
                "type": "listing",
                "source": "Craigslist",
                "url": "https://example.com/list",
                "observed_at": "2026-01-15T00:00:00Z",
                "text": "Listed at $11,400 with 72,000 miles.",
                "facts": {"price": 11400, "mileage": 72000},
                "indirect": False,
            },
            {
                "id": "safe-1",
                "type": "safety",
                "source": "NHTSA",
                "url": "https://www.nhtsa.gov/ratings",
                "observed_at": "2026-01-15T00:00:00Z",
                "text": "NHTSA reports an overall 5-star rating.",
                "facts": {"ncap_stars": 5, "recall_count": 1},
                "indirect": False,
            },
            {
                "id": "rec-1",
                "type": "recall",
                "source": "NHTSA",
                "url": "https://www.nhtsa.gov/recalls",
                "observed_at": "2026-01-15T00:00:00Z",
                "text": "11V123000: SERVICE BRAKES.",
                "facts": {"campaign_number": "11V123000", "component": "SERVICE BRAKES"},
                "indirect": False,
            },
            {
                "id": "rev-1",
                "type": "review",
                "source": "OpinRank",
                "url": "https://example.com/rev",
                "observed_at": "2026-01-15T00:00:00Z",
                "text": "Rear brake pads wore early at 40k miles.",
                "facts": {"review_year": 2009},
                "indirect": False,
            },
            {
                "id": "rev-indirect",
                "type": "review",
                "source": "OpinRank",
                "url": "https://example.com/rev2",
                "observed_at": "2026-01-15T00:00:00Z",
                "text": "Sibling generation review.",
                "facts": {"review_year": 2010},
                "indirect": True,
            },
            {
                "id": "part-1",
                "type": "aftermarket_product",
                "source": "Amazon",
                "url": "https://www.amazon.com/dp/1",
                "observed_at": "2026-01-15T00:00:00Z",
                "text": "Ceramic Brake Pads listed at $64.99; fitment verified.",
                "facts": {"verified_fitment": True, "price": 64.99},
                "indirect": False,
            },
        ],
    }


def test_verify_candidate_k_miles_and_product_prices() -> None:
    candidate = _valid_candidate()
    verify_candidate(candidate)

    # Reject bogus product price
    bad_part_price = copy.deepcopy(candidate)
    bad_part_price["cautions"] = [
        {"text": "Ceramic brake pads listed at $999.", "evidence_ids": ["part-1"]}
    ]
    with pytest.raises(VerificationError):
        verify_candidate(bad_part_price)

    # Listing price remains exact (within $1 rounding tolerance must NOT waive listing price!)
    bad_listing_price = copy.deepcopy(candidate)
    bad_listing_price["pros"] = [
        {"text": "Listed at $11,401 with 72,000 miles.", "evidence_ids": ["list-1"]}
    ]
    with pytest.raises(VerificationError):
        verify_candidate(bad_listing_price)

    # Reject indirect scoring evidence
    bad_indirect = copy.deepcopy(candidate)
    bad_indirect["scoring_evidence_ids"].append("rev-indirect")
    with pytest.raises(VerificationError):
        verify_candidate(bad_indirect)


def test_qwen_think_tag_stripping_and_400_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    class _FakeHTTPResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    def fake_urlopen(req, timeout=120.0):
        body = json.loads(req.data.decode("utf-8"))
        calls.append(body)
        if len(calls) == 1 and "response_format" in body:
            raise HTTPError(req.full_url, 400, "Bad Request", {}, None)
        content = (
            '<think>Internal JSON draft {"wrong": true} end draft.</think>\n'
            "```json\n"
            '{"answer": "ok", "count": 3}\n'
            "```"
        )
        payload = {"choices": [{"message": {"content": content}}]}
        return _FakeHTTPResponse(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr("clickless.llm.qwen.urlopen", fake_urlopen)
    client = QwenClient(base_url="https://mock.local/v1", api_key="test-key")
    result = client.complete_json("Return JSON", "Test prompt")
    assert result == {"answer": "ok", "count": 3}
    assert len(calls) == 2
    assert "response_format" in calls[0]
    assert "response_format" not in calls[1]


def test_apply_grounded_summaries_single_retry_and_fallback() -> None:
    candidate = _valid_candidate()
    template_pros = copy.deepcopy(candidate["pros"])

    class _HallucinatingClient:
        model = "fake-qwen"

        def __init__(self) -> None:
            self.calls = 0

        def complete_json(self, system: str, user: str, max_tokens: int = 1500) -> dict:
            self.calls += 1
            return {
                "candidates": [
                    {
                        "candidate_id": candidate["id"],
                        "strengths": [
                            {
                                "text": "Listed at $99,999 with 72,000 miles.",
                                "supports": [
                                    {
                                        "evidence_id": "list-1",
                                        "quote": "Listed at $11,400 with 72,000 miles.",
                                    }
                                ],
                            }
                        ],
                        "cautions": [],
                    }
                ]
            }

    client = _HallucinatingClient()
    result = apply_grounded_summaries(client, [candidate])
    assert client.calls == 2  # initial call + single retry
    assert result[0]["pros"] == template_pros
    verify_candidate(result[0])


def test_mocked_search_clients_nhtsa_and_max_vehicles_cap(
    mini_catalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Mock Auto.dev
    class _FakeResp:
        def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
            self.status_code = status_code
            self._payload = payload

        def json(self) -> dict[str, Any]:
            return self._payload

    def fake_get(url: str, *args: Any, **kwargs: Any) -> _FakeResp:
        if "api.auto.dev" in url:
            return _FakeResp(
                200,
                {
                    "data": [
                        {
                            "vin": "JF2SH61619H799999",
                            "vehicle": {"year": 2009, "make": "Subaru", "model": "Forester", "trim": "2.5X"},
                            "retailListing": {"price": 10900, "miles": 82000, "city": "Tempe", "state": "AZ", "vdp": "https://auto.dev/vdp/1"},
                        }
                    ]
                },
            )
        if "serpapi.com" in url:
            return _FakeResp(
                200,
                {
                    "shopping_results": [
                        {
                            "product_id": "serp-1",
                            "title": "Subaru Forester Brake Pads",
                            "extracted_price": 59.99,
                            "link": "https://shopping.example.com/item/1",
                            "source": "AutoZone",
                            "rating": 4.7,
                            "reviews": 120,
                        }
                    ]
                },
            )
        raise AssertionError(f"Unexpected URL in fake_get: {url}")

    monkeypatch.setattr(requests, "get", fake_get)
    auto_listings = AutoDevClient("fake-key").search(make="Subaru", model="Forester", year=2009)
    assert len(auto_listings) == 1
    assert auto_listings[0].price == 10900

    serp_offers = SerpApiClient("fake-key").search_products("Subaru Forester brake pads")
    assert len(serp_offers) == 1
    assert serp_offers[0].price == 59.99

    # Mock NHTSA duplicate/missing campaign numbers
    class _FakeUrllibResp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    def fake_nhtsa_urlopen(req, timeout=6.0):
        payload = {
            "results": [
                {"NHTSACampaignNumber": "11V123000", "Component": "BRAKES", "Summary": "S1"},
                {"NHTSACampaignNumber": "11V123000", "Component": "BRAKES", "Summary": "S2"},
                {"NHTSACampaignNumber": "", "Component": "AIR BAGS", "Summary": "S3"},
            ]
        }
        return _FakeUrllibResp(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr("clickless.data.nhtsa.urlopen", fake_nhtsa_urlopen)
    recalls = fetch_recalls("Subaru", "Forester", 2009)
    assert len(recalls) == 3
    assert len({r["id"] for r in recalls}) == 3
    assert len({r["source_id"] for r in recalls}) == 3

    # Test scripts/refresh_retail.py max_vehicles cap on eBay and BestBuy
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "refresh_retail.py"
    spec = importlib.util.spec_from_file_location("refresh_retail", script_path)
    assert spec and spec.loader
    refresh_retail = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(refresh_retail)

    ebay_queries: list[str] = []
    bestbuy_queries: list[str] = []

    class _FakeEbayClient:
        def search(
            self, query: str, limit: int = 10, category_ids: str | None = None
        ) -> list[RetailResult]:
            ebay_queries.append(query)
            return [
                RetailResult(
                    source="eBay",
                    title=f"{query} listing",
                    url="https://www.ebay.com/itm/999",
                    price=11200.0,
                    identifier="v1|999|0",
                    attributes={"mileage": 78000, "location": "Phoenix, AZ"},
                )
            ]

    class _FakeBestBuyClient:
        def search(self, query: str, limit: int = 10) -> list[RetailResult]:
            bestbuy_queries.append(query)
            return [
                RetailResult(
                    source="Best Buy",
                    title=f"{query} dash cam",
                    url="https://www.bestbuy.com/site/1.p",
                    price=149.99,
                    identifier="sku-1",
                    attributes={"rating": 4.5},
                )
            ]

    monkeypatch.setattr(refresh_retail.EbayClient, "from_env", lambda: _FakeEbayClient())
    monkeypatch.setattr(refresh_retail.BestBuyClient, "from_env", lambda: _FakeBestBuyClient())

    cat_copy = copy.deepcopy(mini_catalog)
    ebay_added = refresh_retail.refresh_ebay(cat_copy, per_vehicle=1, max_vehicles=1)
    bestbuy_added = refresh_retail.refresh_bestbuy(cat_copy, per_vehicle=1, max_vehicles=1)
    assert len(ebay_queries) == 1
    assert len(bestbuy_queries) == 1
    assert ebay_added == 1
    assert bestbuy_added == 1


def test_flask_endpoints_and_error_codes(
    use_mini_catalog, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(retrieval_mod, "_DENSE_MODEL", _FakeDense())
    app = create_app()
    client = app.test_client()

    health_resp = client.get("/api/health")
    assert health_resp.status_code == 200
    assert health_resp.get_json() == {"status": "ok"}

    index_resp = client.get("/")
    assert index_resp.status_code == 200
    index_html = index_resp.get_data(as_text=True)
    assert 'id="settings-dialog"' in index_html
    assert 'class="live-check"' in index_html
    assert 'class="gauge"' in index_html

    data_resp = client.get("/api/data-status")
    assert data_resp.status_code == 200
    data_payload = data_resp.get_json()
    assert data_payload["ready"] is True
    assert data_payload["counts"]["vehicles"] == 4
    assert "graph_backend" in data_payload
    assert "retrieval_mode" in data_payload

    model_resp = client.get("/api/model-status")
    assert model_resp.status_code == 200
    assert model_resp.get_json()["available"] is False

    # 400 validations on /api/recommend
    assert client.post("/api/recommend", json={"query": ""}).status_code == 400
    assert client.post("/api/recommend", json={"query": "x" * 1001}).status_code == 400
    assert client.post("/api/recommend", json={"query": "SUV under $20k", "configuration": "Z"}).status_code == 400
    assert client.post("/api/recommend", json={"query": "SUV under $20k", "candidate_limit": 0}).status_code == 400
    assert client.post("/api/recommend", json={"query": "SUV under $20k", "candidate_limit": 11}).status_code == 400

    ok_resp = client.post(
        "/api/recommend",
        json={"query": "Find a Subaru SUV under $20,000", "use_model": False, "configuration": "C", "candidate_limit": 2},
    )
    assert ok_resp.status_code == 200
    ok_json = ok_resp.get_json()
    assert len(ok_json["candidates"]) == 2
    assert "navigation_plan" in ok_json
    assert "cohort_retention" in ok_json
    assert "aspect_bundles" in ok_json["candidates"][0]
    assert "safety_basis" in ok_json["candidates"][0]

    # /api/live-check validation and mocked sources
    assert client.post("/api/live-check", json={"vehicle_id": "missing"}).status_code == 404
    no_keys_live = client.post(
        "/api/live-check", json={"vehicle_id": "subaru-forester-2009-awd"}
    )
    assert no_keys_live.status_code == 200
    assert no_keys_live.get_json()["message"] == "No live marketplace keys are configured."

    from clickless.sources.base import VehicleListing

    class _MockAutoDev:
        def search(self, make: str, model: str, year: int, limit: int = 5):
            return [
                VehicleListing(
                    source="Auto.dev",
                    identifier="vin-1",
                    year=year,
                    make=make,
                    model=model,
                    trim="2.5X",
                    price=11900,
                    mileage=69000,
                    location="Tempe, AZ",
                    vin="JF2SH61619H799999",
                    url="https://auto.dev/vdp/live-1",
                    captured_at="2026-02-01T00:00:00Z",
                    attributes={},
                )
            ]

    class _MockEbay:
        def search(self, query: str, limit: int = 15, category_ids: str | None = None):
            return [
                RetailResult(
                    source="eBay",
                    title="2009 Subaru Forester AWD",
                    url="https://www.ebay.com/itm/live-1",
                    price=11250.0,
                    identifier="ebay-live-1",
                    attributes={"mileage": 74000},
                )
            ]

    monkeypatch.setattr("clickless.web.AutoDevClient.from_env", lambda: _MockAutoDev())
    monkeypatch.setattr("clickless.web.EbayClient.from_env", lambda: _MockEbay())
    mocked_live = client.post(
        "/api/live-check", json={"vehicle_id": "subaru-forester-2009-awd"}
    )
    assert mocked_live.status_code == 200
    mocked_payload = mocked_live.get_json()
    assert mocked_payload["sources"] == ["Auto.dev", "eBay"]
    assert len(mocked_payload["listings"]) == 2
    assert mocked_payload["listings"][0]["price"] == 11250.0

    # 500 when VerificationError raised in recommend
    def raise_verification(*args, **kwargs):
        raise VerificationError("Broken citation")

    monkeypatch.setattr("clickless.web.recommend", raise_verification)
    err_500 = client.post("/api/recommend", json={"query": "SUV under $20k", "use_model": False})
    assert err_500.status_code == 500

    # 503 when CatalogError raised (missing catalog file)
    monkeypatch.setenv("CLICKLESS_CATALOG_PATH", str(tmp_path / "does_not_exist.json"))
    assert client.get("/api/data-status").status_code == 503
    assert (
        client.post(
            "/api/live-check", json={"vehicle_id": "subaru-forester-2009-awd"}
        ).status_code
        == 503
    )

    from clickless.service import recommend as real_recommend

    monkeypatch.setattr("clickless.web.recommend", real_recommend)
    err_503 = client.post("/api/recommend", json={"query": "SUV under $20k", "use_model": False})
    assert err_503.status_code == 503
