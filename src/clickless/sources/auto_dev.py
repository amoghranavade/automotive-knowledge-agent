from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import requests

from .base import SourceError, VehicleListing


class AutoDevClient:
    ENDPOINT = "https://api.auto.dev/listings"

    def __init__(self, api_key: str, timeout: float = 20.0) -> None:
        self.api_key = api_key
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "AutoDevClient | None":
        key = os.getenv("AUTO_DEV_API_KEY", "").strip()
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
                    "vehicle.make": make,
                    "vehicle.model": model,
                    "vehicle.year": year,
                    "retailListing.used": "true",
                    "sort": "price.asc",
                    "limit": min(max(limit, 1), 20),
                },
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Accept": "application/json",
                },
                timeout=self.timeout,
            )
        except requests.RequestException as error:
            raise SourceError(f"Auto.dev request failed: {error}") from error
        if response.status_code >= 400:
            raise SourceError(f"Auto.dev returned HTTP {response.status_code}")
        captured_at = datetime.now(timezone.utc).isoformat()
        listings = []
        for item in response.json().get("data", []):
            parsed = _listing(item, captured_at)
            if parsed is not None:
                listings.append(parsed)
        return listings


def _listing(item: dict[str, Any], captured_at: str) -> VehicleListing | None:
    vehicle = item.get("vehicle") or {}
    retail = item.get("retailListing") or {}
    price = _integer(retail.get("price"))
    mileage = _integer(retail.get("miles") or retail.get("mileage"))
    year = _integer(vehicle.get("year"))
    make = str(vehicle.get("make") or "").strip()
    model = str(vehicle.get("model") or "").strip()
    url = str(retail.get("vdp") or retail.get("url") or "").strip()
    identifier = str(item.get("vin") or vehicle.get("vin") or item.get("@id") or "").strip()
    if None in {price, mileage, year} or not all((make, model, url, identifier)):
        return None
    if not 500 <= price <= 250_000 or not 0 <= mileage <= 500_000:
        return None
    location_value = item.get("location")
    location = str(retail.get("city") or retail.get("state") or "")
    if not location and isinstance(location_value, list) and len(location_value) == 2:
        location = f"{location_value[1]}, {location_value[0]}"
    return VehicleListing(
        source="Auto.dev",
        identifier=identifier,
        year=year,
        make=make,
        model=model,
        trim=str(vehicle.get("trim") or ""),
        price=price,
        mileage=mileage,
        location=location,
        vin=str(item.get("vin") or vehicle.get("vin") or ""),
        url=url,
        captured_at=str(item.get("updatedAt") or captured_at),
        attributes={"dealer_id": retail.get("dealerId")},
    )


def _integer(value: object) -> int | None:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None
