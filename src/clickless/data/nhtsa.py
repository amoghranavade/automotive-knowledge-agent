from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


ENDPOINT = "https://api.nhtsa.gov/recalls/recallsByVehicle"
RATINGS_ENDPOINT = "https://api.nhtsa.gov/SafetyRatings"


class NHTSAError(RuntimeError):
    pass


def fetch_recalls(make: str, model: str, year: int, timeout: float = 6.0) -> list[dict[str, Any]]:
    query = urlencode({"make": make, "model": model, "modelYear": year})
    request = Request(f"{ENDPOINT}?{query}", headers={"User-Agent": "ClickLessAutomotive/1.0"})
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise NHTSAError(f"NHTSA request failed: {exc}") from exc

    results = payload.get("results")
    if not isinstance(results, list):
        raise NHTSAError("NHTSA returned an invalid response")
    return [_normalize_recall(item) for item in results]


def fetch_safety_rating(
    make: str,
    model: str,
    year: int,
    timeout: float = 8.0,
) -> dict[str, Any] | None:
    lookup_url = (
        f"{RATINGS_ENDPOINT}/modelyear/{year}/make/{quote(make)}/"
        f"model/{quote(model)}?format=json"
    )
    lookup = _get_json(lookup_url, timeout)
    matches = lookup.get("Results") or lookup.get("results") or []
    if not isinstance(matches, list) or not matches:
        return None
    vehicle_id = matches[0].get("VehicleId") or matches[0].get("vehicleId")
    if not vehicle_id:
        return None
    detail = _get_json(f"{RATINGS_ENDPOINT}/VehicleId/{vehicle_id}?format=json", timeout)
    records = detail.get("Results") or detail.get("results") or []
    if not isinstance(records, list) or not records:
        return None
    record = records[0]
    rating = _rating(record.get("OverallRating"))
    return {
        "vehicle_id": str(vehicle_id),
        "overall_stars": rating,
        "frontal_stars": _rating(record.get("OverallFrontCrashRating")),
        "side_stars": _rating(record.get("OverallSideCrashRating")),
        "rollover_stars": _rating(record.get("RolloverRating")),
        "source_url": f"https://www.nhtsa.gov/vehicle/{year}/{quote(make)}/{quote(model)}",
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }


def _get_json(url: str, timeout: float) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": "ClickLessAutomotive/1.0"})
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise NHTSAError(f"NHTSA request failed: {exc}") from exc
    if not isinstance(payload, dict):
        raise NHTSAError("NHTSA returned an invalid response")
    return payload


def _rating(value: Any) -> int | None:
    try:
        rating = int(float(value))
    except (TypeError, ValueError):
        return None
    return rating if 0 <= rating <= 5 else None


def recalls_with_fallback(
    catalog: dict[str, Any],
    vehicle: dict[str, Any],
    live: bool = False,
    timeout: float = 6.0,
) -> tuple[list[dict[str, Any]], str]:
    if live:
        try:
            recalls = fetch_recalls(vehicle["make"], vehicle["model"], vehicle["year"], timeout)
            return recalls, "live"
        except NHTSAError:
            pass
    fixtures = [item for item in catalog["recalls"] if item["vehicle_id"] == vehicle["id"]]
    return fixtures, "snapshot"


def _normalize_recall(item: dict[str, Any]) -> dict[str, Any]:
    campaign = str(item.get("NHTSACampaignNumber") or "")
    return {
        "id": f"nhtsa-{campaign}" if campaign else "nhtsa-unknown",
        "campaign_number": campaign,
        "component": item.get("Component") or "unknown",
        "summary": item.get("Summary") or "",
        "consequence": item.get("Consequence") or "",
        "remedy": item.get("Remedy") or "",
        "manufacturer": item.get("Manufacturer") or "",
        "report_received_date": item.get("ReportReceivedDate") or "",
        "source_name": "NHTSA",
        "source_url": "https://www.nhtsa.gov/recalls",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "source_id": f"nhtsa-{campaign}" if campaign else "nhtsa-unknown",
    }
