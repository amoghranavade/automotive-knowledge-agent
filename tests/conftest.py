from __future__ import annotations

import copy
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

import pytest
import requests

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


BLOCKED_EXTERNAL_HOSTS = (
    "api.auto.dev",
    "api.marketcheck.com",
    "serpapi.com",
    "api.ebay.com",
    "api.bestbuy.com",
    "api.nhtsa.gov",
    "openai.rc.asu.edu",
)


@pytest.fixture(autouse=True)
def _block_live_search_apis(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure tests never leak paid search API credentials or hit live APIs."""
    for env_var in (
        "AUTO_DEV_API_KEY",
        "MARKETCHECK_API_KEY",
        "SERPAPI_API_KEY",
        "EBAY_CLIENT_ID",
        "EBAY_CLIENT_SECRET",
        "BESTBUY_API_KEY",
        "CLICKLESS_QWEN_API_KEY",
    ):
        monkeypatch.delenv(env_var, raising=False)

    monkeypatch.setenv("CLICKLESS_LIVE_NHTSA", "false")
    monkeypatch.setenv("CLICKLESS_GRAPH_BACKEND", "memory")

    original_session_request = requests.Session.request
    original_urlopen = urllib.request.urlopen

    def guarded_session_request(
        self: requests.Session,
        method: str,
        url: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        lowered = str(url).casefold()
        if any(host in lowered for host in BLOCKED_EXTERNAL_HOSTS):
            raise AssertionError(
                f"Blocked unmocked external HTTP {method} request in tests: {url}"
            )
        return original_session_request(self, method, url, *args, **kwargs)

    def guarded_urlopen(url_or_req: Any, *args: Any, **kwargs: Any) -> Any:
        target = getattr(url_or_req, "full_url", url_or_req)
        lowered = str(target).casefold()
        if any(host in lowered for host in BLOCKED_EXTERNAL_HOSTS):
            raise AssertionError(
                f"Blocked unmocked external urllib request in tests: {target}"
            )
        return original_urlopen(url_or_req, *args, **kwargs)

    monkeypatch.setattr(requests.Session, "request", guarded_session_request)
    monkeypatch.setattr(urllib.request, "urlopen", guarded_urlopen)


def make_mini_catalog() -> dict[str, Any]:
    """Build a deterministic synthetic multi-variant catalog covering all test paths."""
    captured = "2026-01-15T12:00:00Z"
    return copy.deepcopy(
        {
            "snapshot": {
                "id": "mini-snapshot-2026-01-15",
                "created_at": captured,
                "mode": "frozen_snapshot",
                "notice": (
                    "Synthetic test snapshot for offline unit and integration tests."
                ),
                "sources": [
                    "Craigslist snapshot via Kaggle",
                    "UCI OpinRank",
                    "Amazon Reviews 2023 Automotive",
                    "NHTSA",
                ],
            },
            "vehicles": [
                {
                    "id": "subaru-forester-2009-awd",
                    "generation_id": "subaru-forester-gen-3",
                    "year": 2009,
                    "make": "Subaru",
                    "model": "Forester",
                    "trim": "2.5X",
                    "body_type": "SUV",
                    "drivetrain": "AWD",
                },
                {
                    "id": "subaru-forester-2010-awd",
                    "generation_id": "subaru-forester-gen-3",
                    "year": 2010,
                    "make": "Subaru",
                    "model": "Forester",
                    "trim": "2.5X Limited",
                    "body_type": "SUV",
                    "drivetrain": "AWD",
                },
                {
                    "id": "honda-accord-2008-fwd",
                    "generation_id": "honda-accord-gen-8",
                    "year": 2008,
                    "make": "Honda",
                    "model": "Accord",
                    "trim": "EX-L",
                    "body_type": "Sedan",
                    "drivetrain": "FWD",
                },
                {
                    "id": "ford-f-150-2011-4wd",
                    "generation_id": "ford-f-150-gen-12",
                    "year": 2011,
                    "make": "Ford",
                    "model": "F-150",
                    "trim": "XLT",
                    "body_type": "Truck",
                    "drivetrain": "4WD",
                },
            ],
            "listings": [
                {
                    "id": "ebay-listing-v1|123456|0",
                    "vehicle_id": "subaru-forester-2009-awd",
                    "price": 9200,
                    "mileage": 135000,
                    "location": "Tempe, AZ",
                    "vin": "JF2SH61619H700001",
                    "source_id": "ebay-listing-v1|123456|0",
                    "source_name": "eBay Browse API",
                    "source_url": "https://www.ebay.com/itm/123456",
                    "captured_at": captured,
                    "data_mode": "frozen_snapshot",
                },
                {
                    "id": "listing-forester-2009-clean",
                    "vehicle_id": "subaru-forester-2009-awd",
                    "price": 11400,
                    "mileage": 72000,
                    "location": "Phoenix, AZ",
                    "vin": "JF2SH61619H700002",
                    "source_id": "craigslist-forester-2009-clean",
                    "source_name": "Craigslist snapshot via Kaggle",
                    "source_url": "https://phoenix.craigslist.org/cto/d/subaru-forester/1001.html",
                    "captured_at": captured,
                    "data_mode": "frozen_snapshot",
                },
                {
                    "id": "listing-forester-2010-1",
                    "vehicle_id": "subaru-forester-2010-awd",
                    "price": 12800,
                    "mileage": 68000,
                    "location": "Mesa, AZ",
                    "vin": "JF2SH6369AH700003",
                    "source_id": "craigslist-forester-2010-1",
                    "source_name": "Craigslist snapshot via Kaggle",
                    "source_url": "https://phoenix.craigslist.org/cto/d/subaru-forester-2010/1002.html",
                    "captured_at": captured,
                    "data_mode": "frozen_snapshot",
                },
                {
                    "id": "listing-accord-2008-1",
                    "vehicle_id": "honda-accord-2008-fwd",
                    "price": 8900,
                    "mileage": 88000,
                    "location": "Scottsdale, AZ",
                    "vin": "1HGCP26888A000004",
                    "source_id": "craigslist-accord-2008-1",
                    "source_name": "Craigslist snapshot via Kaggle",
                    "source_url": "https://phoenix.craigslist.org/cto/d/honda-accord/1003.html",
                    "captured_at": captured,
                    "data_mode": "frozen_snapshot",
                },
                {
                    "id": "listing-f150-2011-1",
                    "vehicle_id": "ford-f-150-2011-4wd",
                    "price": 19500,
                    "mileage": 92000,
                    "location": "Tucson, AZ",
                    "vin": "1FTFW1ET2BFA00005",
                    "source_id": "craigslist-f150-2011-1",
                    "source_name": "Craigslist snapshot via Kaggle",
                    "source_url": "https://tucson.craigslist.org/cto/d/ford-f150/1004.html",
                    "captured_at": captured,
                    "data_mode": "frozen_snapshot",
                },
            ],
            "safety_ratings": [
                {
                    "id": "nhtsa-rating-forester-2009",
                    "vehicle_id": "subaru-forester-2009-awd",
                    "overall_stars": 5,
                    "frontal_stars": 5,
                    "side_stars": 5,
                    "rollover_stars": 4,
                    "source_id": "nhtsa-rating-forester-2009",
                    "source_name": "NHTSA NCAP",
                    "source_url": "https://www.nhtsa.gov/vehicle/2009/SUBARU/FORESTER",
                    "captured_at": captured,
                },
                {
                    "id": "nhtsa-rating-forester-2010",
                    "vehicle_id": "subaru-forester-2010-awd",
                    "overall_stars": 5,
                    "frontal_stars": 5,
                    "side_stars": 5,
                    "rollover_stars": 4,
                    "source_id": "nhtsa-rating-forester-2010",
                    "source_name": "NHTSA NCAP",
                    "source_url": "https://www.nhtsa.gov/vehicle/2010/SUBARU/FORESTER",
                    "captured_at": captured,
                },
                {
                    "id": "nhtsa-rating-f150-2011",
                    "vehicle_id": "ford-f-150-2011-4wd",
                    "overall_stars": 4,
                    "frontal_stars": 4,
                    "side_stars": 5,
                    "rollover_stars": 3,
                    "source_id": "nhtsa-rating-f150-2011",
                    "source_name": "NHTSA NCAP",
                    "source_url": "https://www.nhtsa.gov/vehicle/2011/FORD/F-150",
                    "captured_at": captured,
                },
            ],
            "recalls": [
                {
                    "id": "subaru-forester-2009-awd-nhtsa-11V123000",
                    "vehicle_id": "subaru-forester-2009-awd",
                    "campaign_number": "11V123000",
                    "component": "SERVICE BRAKES, HYDRAULIC",
                    "summary": "Brake line corrosion can increase stopping distance.",
                    "consequence": "Extended stopping distance may increase crash risk.",
                    "remedy": "Dealers will inspect and replace affected brake lines.",
                    "manufacturer": "Subaru of America, Inc.",
                    "report_received_date": "2011-03-10",
                    "source_id": "nhtsa-11V123000",
                    "source_name": "NHTSA",
                    "source_url": "https://www.nhtsa.gov/recalls",
                    "captured_at": captured,
                },
                {
                    "id": "honda-accord-2008-fwd-nhtsa-16V344000",
                    "vehicle_id": "honda-accord-2008-fwd",
                    "campaign_number": "16V344000",
                    "component": "AIR BAGS",
                    "summary": "Passenger frontal airbag inflator may rupture upon deployment.",
                    "consequence": "Metal fragments may strike occupants.",
                    "remedy": "Dealers will replace the airbag inflator free of charge.",
                    "manufacturer": "American Honda Motor Co.",
                    "report_received_date": "2016-05-23",
                    "source_id": "nhtsa-16V344000",
                    "source_name": "NHTSA",
                    "source_url": "https://www.nhtsa.gov/recalls",
                    "captured_at": captured,
                },
            ],
            "reviews": [
                {
                    "id": "opinrank-forester-2009-0",
                    "vehicle_id": "subaru-forester-2009-awd",
                    "review_year": 2009,
                    "text": (
                        "Engine is dependable in snow and cold mornings. "
                        "Rear brake pads wore early at 40k miles and required costly service."
                    ),
                    "rating": None,
                    "match_type": "exact_year",
                    "source_kind": "owner_review",
                    "aspects": [
                        {
                            "id": "opinrank-forester-2009-0-aspect-0",
                            "aspect": "reliability",
                            "subsystem": "engine",
                            "polarity": "positive",
                            "span": "Engine is dependable in snow and cold mornings.",
                            "method": "lexicon",
                        },
                        {
                            "id": "opinrank-forester-2009-0-aspect-1",
                            "aspect": "maintenance",
                            "subsystem": "brakes",
                            "polarity": "negative",
                            "span": "Rear brake pads wore early at 40k miles and required costly service.",
                            "method": "lexicon",
                        },
                    ],
                    "source_id": "opinrank-forester-2009-0",
                    "source_name": "Edmunds OpinRank",
                    "source_url": "https://archive.ics.uci.edu/dataset/205/opinrank+review+dataset",
                    "captured_at": captured,
                    "data_mode": "frozen_snapshot",
                },
                {
                    "id": "opinrank-forester-2009-1",
                    "vehicle_id": "subaru-forester-2009-awd",
                    "review_year": 2009,
                    "text": (
                        "Cargo space easily fits family luggage. "
                        "Good highway mpg, but city fuel economy drops in winter. "
                        "Scheduled oil service happens every 6,000 miles."
                    ),
                    "rating": None,
                    "match_type": "exact_year",
                    "source_kind": "owner_review",
                    "aspects": [
                        {
                            "id": "opinrank-forester-2009-1-aspect-0",
                            "aspect": "cargo",
                            "subsystem": "cabin",
                            "polarity": "positive",
                            "span": "Cargo space easily fits family luggage.",
                            "method": "lexicon",
                        },
                        {
                            "id": "opinrank-forester-2009-1-aspect-1",
                            "aspect": "fuel_economy",
                            "subsystem": "fuel_system",
                            "polarity": "mixed",
                            "span": "Good highway mpg, but city fuel economy drops in winter.",
                            "method": "lexicon",
                        },
                        {
                            "id": "opinrank-forester-2009-1-aspect-2",
                            "aspect": "maintenance",
                            "subsystem": "engine",
                            "polarity": "neutral",
                            "span": "Scheduled oil service happens every 6,000 miles.",
                            "method": "lexicon",
                        },
                    ],
                    "source_id": "opinrank-forester-2009-1",
                    "source_name": "Edmunds OpinRank",
                    "source_url": "https://archive.ics.uci.edu/dataset/205/opinrank+review+dataset",
                    "captured_at": captured,
                    "data_mode": "frozen_snapshot",
                },
                {
                    "id": "opinrank-forester-2009-2",
                    "vehicle_id": "subaru-forester-2009-awd",
                    "review_year": 2009,
                    "text": (
                        "AWD traction is durable and reliable after 90,000 miles. "
                        "Affordable routine maintenance keeps ownership simple."
                    ),
                    "rating": None,
                    "match_type": "exact_year",
                    "source_kind": "owner_review",
                    "aspects": [
                        {
                            "id": "opinrank-forester-2009-2-aspect-0",
                            "aspect": "reliability",
                            "subsystem": "drivetrain",
                            "polarity": "positive",
                            "span": "AWD traction is durable and reliable after 90,000 miles.",
                            "method": "lexicon",
                        },
                        {
                            "id": "opinrank-forester-2009-2-aspect-1",
                            "aspect": "maintenance",
                            "subsystem": "unknown",
                            "polarity": "positive",
                            "span": "Affordable routine maintenance keeps ownership simple.",
                            "method": "lexicon",
                        },
                    ],
                    "source_id": "opinrank-forester-2009-2",
                    "source_name": "Edmunds OpinRank",
                    "source_url": "https://archive.ics.uci.edu/dataset/205/opinrank+review+dataset",
                    "captured_at": captured,
                    "data_mode": "frozen_snapshot",
                },
                {
                    "id": "opinrank-forester-2010-0",
                    "vehicle_id": "subaru-forester-2010-awd",
                    "review_year": 2010,
                    "text": "Smooth highway ride and dependable engine in winter.",
                    "rating": None,
                    "match_type": "exact_year",
                    "source_kind": "owner_review",
                    "aspects": [
                        {
                            "id": "opinrank-forester-2010-0-aspect-0",
                            "aspect": "reliability",
                            "subsystem": "engine",
                            "polarity": "positive",
                            "span": "Smooth highway ride and dependable engine in winter.",
                            "method": "lexicon",
                        }
                    ],
                    "source_id": "opinrank-forester-2010-0",
                    "source_name": "Edmunds OpinRank",
                    "source_url": "https://archive.ics.uci.edu/dataset/205/opinrank+review+dataset",
                    "captured_at": captured,
                    "data_mode": "frozen_snapshot",
                },
                {
                    "id": "opinrank-forester-2010-1",
                    "vehicle_id": "subaru-forester-2010-awd",
                    "review_year": 2010,
                    "text": "Brake rotor replacement cost more than expected.",
                    "rating": None,
                    "match_type": "exact_year",
                    "source_kind": "owner_review",
                    "aspects": [
                        {
                            "id": "opinrank-forester-2010-1-aspect-0",
                            "aspect": "maintenance",
                            "subsystem": "brakes",
                            "polarity": "negative",
                            "span": "Brake rotor replacement cost more than expected.",
                            "method": "lexicon",
                        }
                    ],
                    "source_id": "opinrank-forester-2010-1",
                    "source_name": "Edmunds OpinRank",
                    "source_url": "https://archive.ics.uci.edu/dataset/205/opinrank+review+dataset",
                    "captured_at": captured,
                    "data_mode": "frozen_snapshot",
                },
                {
                    "id": "opinrank-accord-2008-0",
                    "vehicle_id": "honda-accord-2008-fwd",
                    "review_year": 2008,
                    "text": "Great gas mileage and reliable daily commuter.",
                    "rating": None,
                    "match_type": "exact_year",
                    "source_kind": "owner_review",
                    "aspects": [
                        {
                            "id": "opinrank-accord-2008-0-aspect-0",
                            "aspect": "fuel_economy",
                            "subsystem": "fuel_system",
                            "polarity": "positive",
                            "span": "Great gas mileage and reliable daily commuter.",
                            "method": "lexicon",
                        },
                        {
                            "id": "opinrank-accord-2008-0-aspect-1",
                            "aspect": "reliability",
                            "subsystem": "engine",
                            "polarity": "positive",
                            "span": "Great gas mileage and reliable daily commuter.",
                            "method": "lexicon",
                        },
                    ],
                    "source_id": "opinrank-accord-2008-0",
                    "source_name": "Edmunds OpinRank",
                    "source_url": "https://archive.ics.uci.edu/dataset/205/opinrank+review+dataset",
                    "captured_at": captured,
                    "data_mode": "frozen_snapshot",
                },
                {
                    "id": "amazon-review-accord-rotors",
                    "vehicle_id": "honda-accord-2008-fwd",
                    "review_year": 2008,
                    "text": "Affordable brake rotor kit, easy maintenance install.",
                    "rating": 5.0,
                    "match_type": "product_fitment_unverified",
                    "source_kind": "product_review",
                    "aspects": [
                        {
                            "id": "amazon-review-accord-rotors-aspect-0",
                            "aspect": "maintenance",
                            "subsystem": "brakes",
                            "polarity": "positive",
                            "span": "Affordable brake rotor kit, easy maintenance install.",
                            "method": "lexicon",
                        }
                    ],
                    "source_id": "amazon-review-accord-rotors",
                    "source_name": "Amazon Reviews 2023",
                    "source_url": "https://www.amazon.com/dp/B00ACCORD1",
                    "captured_at": captured,
                    "data_mode": "frozen_snapshot",
                },
                {
                    "id": "opinrank-f150-2011-0",
                    "vehicle_id": "ford-f-150-2011-4wd",
                    "review_year": 2011,
                    "text": (
                        "Dependable truck bed cargo capacity, though poor gas mileage in city traffic."
                    ),
                    "rating": None,
                    "match_type": "exact_year",
                    "source_kind": "owner_review",
                    "aspects": [
                        {
                            "id": "opinrank-f150-2011-0-aspect-0",
                            "aspect": "cargo",
                            "subsystem": "body",
                            "polarity": "positive",
                            "span": "Dependable truck bed cargo capacity",
                            "method": "lexicon",
                        },
                        {
                            "id": "opinrank-f150-2011-0-aspect-1",
                            "aspect": "fuel_economy",
                            "subsystem": "fuel_system",
                            "polarity": "negative",
                            "span": "poor gas mileage in city traffic.",
                            "method": "lexicon",
                        },
                        {
                            "id": "opinrank-f150-2011-0-aspect-2",
                            "aspect": "reliability",
                            "subsystem": "body",
                            "polarity": "positive",
                            "span": "Dependable truck bed cargo capacity",
                            "method": "lexicon",
                        },
                    ],
                    "source_id": "opinrank-f150-2011-0",
                    "source_name": "Edmunds OpinRank",
                    "source_url": "https://archive.ics.uci.edu/dataset/205/opinrank+review+dataset",
                    "captured_at": captured,
                    "data_mode": "frozen_snapshot",
                },
            ],
            "parts": [
                {
                    "id": "amazon-product-forester-brakes",
                    "vehicle_id": "subaru-forester-2009-awd",
                    "name": "Subaru Forester Ceramic Brake Pads",
                    "price": 64.99,
                    "subsystem": "brakes",
                    "verified_fitment": True,
                    "rating": 4.7,
                    "source_id": "amazon-meta-forester-brakes",
                    "source_name": "Amazon Reviews 2023 metadata",
                    "source_url": "https://www.amazon.com/dp/B00FORBRK1",
                    "captured_at": captured,
                    "data_mode": "frozen_snapshot",
                },
                {
                    "id": "amazon-product-universal-battery",
                    "vehicle_id": "subaru-forester-2009-awd",
                    "name": "Universal AGM Replacement Battery",
                    "price": 189.50,
                    "subsystem": "battery",
                    "verified_fitment": False,
                    "rating": 4.4,
                    "source_id": "amazon-meta-universal-battery",
                    "source_name": "Amazon Reviews 2023 metadata",
                    "source_url": "https://www.amazon.com/dp/B00UNIBAT1",
                    "captured_at": captured,
                    "data_mode": "frozen_snapshot",
                },
                {
                    "id": "amazon-product-accord-rotors",
                    "vehicle_id": "honda-accord-2008-fwd",
                    "name": "Honda Accord Brake Rotor Kit",
                    "price": 119.95,
                    "subsystem": "brakes",
                    "verified_fitment": True,
                    "rating": 4.6,
                    "source_id": "amazon-meta-accord-rotors",
                    "source_name": "Amazon Reviews 2023 metadata",
                    "source_url": "https://www.amazon.com/dp/B00ACCORD1",
                    "captured_at": captured,
                    "data_mode": "frozen_snapshot",
                },
                {
                    "id": "amazon-product-f150-battery",
                    "vehicle_id": "ford-f-150-2011-4wd",
                    "name": "Ford F-150 Heavy Duty Battery",
                    "price": 179.00,
                    "subsystem": "battery",
                    "verified_fitment": True,
                    "rating": 4.8,
                    "source_id": "amazon-meta-f150-battery",
                    "source_name": "Amazon Reviews 2023 metadata",
                    "source_url": "https://www.amazon.com/dp/B00F150BAT",
                    "captured_at": captured,
                    "data_mode": "frozen_snapshot",
                },
            ],
        }
    )


@pytest.fixture
def mini_catalog() -> dict[str, Any]:
    return make_mini_catalog()


@pytest.fixture
def catalog_path(tmp_path: Path, mini_catalog: dict[str, Any]) -> Path:
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(mini_catalog, indent=2), encoding="utf-8")
    return path


@pytest.fixture
def use_mini_catalog(
    monkeypatch: pytest.MonkeyPatch, catalog_path: Path
) -> Path:
    monkeypatch.setenv("CLICKLESS_CATALOG_PATH", str(catalog_path))
    return catalog_path
