from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import requests

import clickless.config  # noqa: F401

from .base import SourceError, VehicleListing


class MarketCheckClient:
    ENDPOINT = "https://api.marketcheck.com/v2/search/car/active"

    def __init__(self, api_key: str, timeout: float = 20.0) -> None:
        self.api_key = api_key
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "MarketCheckClient | None":
        key = os.getenv("MARKETCHECK_API_KEY", "").strip()
        return cls(key) if key else None

    def search(
        self,
        *,
        make: str,
        model: str,
        year: int,
        limit: int = 5,
    ) -> list[VehicleListing]:
        try:
            response = requests.get(
                self.ENDPOINT,
                params={
                    "api_key": self.api_key,
                    "car_type": "used",
                    "make": make,
                    "model": model,
                    "year": year,
                    "rows": min(max(limit, 1), 50),
                    "has_price": "true",
                    "has_miles": "true",
                    "sort_by": "price",
                    "sort_order": "asc",
                },
                headers={"Accept": "application/json"},
                timeout=self.timeout,
            )
        except requests.RequestException as error:
            raise SourceError(f"MarketCheck request failed: {error}") from error
        if response.status_code >= 400:
            raise SourceError(f"MarketCheck returned HTTP {response.status_code}")
        captured_at = datetime.now(timezone.utc).isoformat()
        listings = []
        for item in response.json().get("listings", []):
            parsed = _listing(item, captured_at)
            if parsed is not None:
                listings.append(parsed)
        return listings


def _listing(item: dict[str, Any], captured_at: str) -> VehicleListing | None:
    build = item.get("build") or {}
    dealer = item.get("dealer") or {}
    price = _integer(item.get("price"))
    mileage = _integer(item.get("miles"))
    year = _integer(build.get("year") or item.get("year"))
    make = str(build.get("make") or item.get("make") or "").strip()
    model = str(build.get("model") or item.get("model") or "").strip()
    url = str(item.get("vdp_url") or "").strip()
    identifier = str(item.get("id") or item.get("vin") or "").strip()
    if None in {price, mileage, year} or not all((make, model, url, identifier)):
        return None
    if not 500 <= price <= 250_000 or not 0 <= mileage <= 500_000:
        return None
    location = ", ".join(
        value
        for value in (
            str(dealer.get("city") or "").strip(),
            str(dealer.get("state") or "").strip(),
        )
        if value
    )
    return VehicleListing(
        source="MarketCheck",
        identifier=identifier,
        year=year,
        make=make,
        model=model,
        trim=str(build.get("trim") or item.get("trim") or ""),
        price=price,
        mileage=mileage,
        location=location,
        vin=str(item.get("vin") or ""),
        url=url,
        captured_at=str(item.get("last_seen_at_date") or captured_at),
        attributes={
            "dealer": dealer.get("name"),
            "source_domain": item.get("source"),
            "days_on_market": item.get("dom_active"),
        },
    )


def _integer(value: object) -> int | None:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None
