from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


ENDPOINT = "https://api.nhtsa.gov/recalls/recallsByVehicle"
RATINGS_ENDPOINT = "https://api.nhtsa.gov/SafetyRatings"


class NHTSAError(RuntimeError):
    pass


# Catalog model strings that NHTSA files under another make or name.
MODEL_ALIASES = {
    ("ram", "dr"): "1500",  # "DR" is the 2002-2008 Ram 1500 platform code
}


def model_variants(model: str) -> list[str]:
    """Craigslist model strings often carry a trim ("cr-v ex-l"); NHTSA wants the base model."""
    tokens = str(model or "").split()
    variants: list[str] = []
    for size in range(len(tokens), 0, -1):
        name = " ".join(tokens[:size])
        hyphenated = re.sub(r"\b([a-z]+)(\d{2,4})\b", r"\1-\2", name)  # f250 -> f-250
        compact = name.replace("-", "")
        joined = re.sub(r"\b([a-z]+) (\d{2,4})\b", r"\1\2", name)  # es 350 -> es350
        for base in (name, hyphenated, compact, joined):
            variants.append(base)
            if "super duty" in base:
                variants.append(base.replace("super duty", "sd"))
            if re.fullmatch(r"f-?[2-5]50", base):
                variants.append(f"{base} sd")  # NHTSA files Super Duty pickups as "F-250 SD"
    return list(dict.fromkeys(item.strip() for item in variants if item.strip()))


def lookup_names(make: str, model: str, year: int) -> list[tuple[str, str]]:
    """(make, model) pairs to try, most specific first."""
    make = str(make or "").strip().lower()
    model = MODEL_ALIASES.get((make, str(model or "").strip().lower()), model)
    pairs = [(make, name) for name in model_variants(model)]
    if make == "ram" and int(year) < 2011:
        # Ram became its own brand for model year 2011; earlier trucks are "Dodge Ram 2500".
        pairs = [("dodge", f"ram {name}") for name in model_variants(model)] + pairs
    return list(dict.fromkeys(pairs))


def fetch_recalls(make: str, model: str, year: int, timeout: float = 6.0) -> list[dict[str, Any]]:
    """Try each name variant; a 400 for one spelling must not stop the others."""
    errors = []
    for lookup_make, name in lookup_names(make, model, year):
        try:
            results = _fetch_recalls_once(lookup_make, name, year, timeout)
        except NHTSAError as error:
            errors.append(error)
            continue
        if results:
            return results
    if errors and len(errors) == len(lookup_names(make, model, year)):
        raise errors[-1]
    return []


def _fetch_recalls_once(make: str, model: str, year: int, timeout: float) -> list[dict[str, Any]]:
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
    seen_ids: set[str] = set()
    return [
        _normalize_recall(item, index=index, seen_ids=seen_ids)
        for index, item in enumerate(results)
    ]


def fetch_safety_rating(
    make: str,
    model: str,
    year: int,
    timeout: float = 8.0,
    drivetrain: str | None = None,
) -> dict[str, Any] | None:
    matches: list[dict[str, Any]] = []
    for lookup_make, name in lookup_names(make, model, year):
        lookup_url = (
            f"{RATINGS_ENDPOINT}/modelyear/{year}/make/{quote(lookup_make)}/"
            f"model/{quote(name)}?format=json"
        )
        try:
            lookup = _get_json(lookup_url, timeout)
        except NHTSAError:
            continue  # this spelling is unknown to NHTSA; try the next one
        found = lookup.get("Results") or lookup.get("results") or []
        if isinstance(found, list) and found:
            matches = found
            make, model = lookup_make, name
            break
    if not matches:
        return None
    chosen = _match_drivetrain(matches, drivetrain)
    vehicle_id = chosen.get("VehicleId") or chosen.get("vehicleId")
    if not vehicle_id:
        return None
    detail = _get_json(f"{RATINGS_ENDPOINT}/VehicleId/{vehicle_id}?format=json", timeout)
    records = detail.get("Results") or detail.get("results") or []
    if not isinstance(records, list) or not records:
        return None
    record = records[0]
    frontal = _rating(record.get("OverallFrontCrashRating")) or _mean_rating(
        record.get("FrontCrashDriversideRating"),
        record.get("FrontCrashPassengersideRating"),
    )
    side = _rating(record.get("OverallSideCrashRating")) or _mean_rating(
        record.get("SideCrashDriversideRating"),
        record.get("SideCrashPassengersideRating"),
        record.get("SidePoleCrashRating"),
    )
    rollover = _rating(record.get("RolloverRating")) or _rating(
        record.get("RolloverRating2")
    )
    overall = _rating(record.get("OverallRating"))
    return {
        "vehicle_id": str(vehicle_id),
        "description": record.get("VehicleDescription", ""),
        "overall_stars": overall,
        "frontal_stars": frontal,
        "side_stars": side,
        "rollover_stars": rollover,
        # Pre-2011 models are "Not Rated" overall but keep these per-seat scores.
        "front_driver_stars": _rating(record.get("FrontCrashDriversideRating")),
        "front_passenger_stars": _rating(record.get("FrontCrashPassengersideRating")),
        "side_driver_stars": _rating(record.get("SideCrashDriversideRating")),
        "side_passenger_stars": _rating(record.get("SideCrashPassengersideRating")),
        "source_url": f"https://www.nhtsa.gov/vehicle/{year}/{quote(make)}/{quote(model)}",
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }


DRIVETRAIN_TOKENS = {
    "AWD": ("AWD", "4WD"),
    "4WD": ("4WD", "AWD"),
    "FWD": ("FWD", "2WD"),
    "RWD": ("RWD", "2WD"),
}


def _match_drivetrain(matches: list[dict[str, Any]], drivetrain: str | None) -> dict[str, Any]:
    for token in DRIVETRAIN_TOKENS.get(str(drivetrain or "").upper(), ()):
        for item in matches:
            description = str(item.get("VehicleDescription") or "").upper()
            if token in description.split() or f" {token}" in description:
                return item
    return matches[0]


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


TEST_FIELDS = (
    "frontal_stars",
    "side_stars",
    "rollover_stars",
    "front_driver_stars",
    "front_passenger_stars",
    "side_driver_stars",
    "side_passenger_stars",
)


def safety_stars(record: dict[str, Any] | None) -> tuple[float | int | None, str]:
    """Overall NCAP stars when NHTSA has them, else the average of the individual tests."""
    if not record:
        return None, "none"
    if record.get("overall_stars") is not None:
        return record["overall_stars"], "overall"
    values = [record[field] for field in TEST_FIELDS if record.get(field) is not None]
    if not values:
        return None, "none"
    return round(sum(values) / len(values), 1), "test_average"


def _rating(value: Any) -> int | None:
    try:
        rating = int(float(value))
    except (TypeError, ValueError):
        return None
    return rating if 1 <= rating <= 5 else None


def _mean_rating(*values: Any) -> int | None:
    valid = [parsed for parsed in (_rating(value) for value in values) if parsed is not None]
    if not valid:
        return None
    return int(round(sum(valid) / len(valid)))


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


def _normalize_recall(
    item: dict[str, Any],
    index: int | None = None,
    seen_ids: set[str] | None = None,
) -> dict[str, Any]:
    campaign = str(item.get("NHTSACampaignNumber") or "").strip()
    if campaign:
        base_id = f"nhtsa-{campaign}"
    elif index is not None:
        base_id = f"nhtsa-unknown-{index}"
    else:
        base_id = "nhtsa-unknown"

    record_id = base_id
    if seen_ids is not None and record_id in seen_ids:
        suffix = index if index is not None else len(seen_ids)
        record_id = f"{base_id}-{suffix}"
        while record_id in seen_ids:
            suffix += 1
            record_id = f"{base_id}-{suffix}"
    if seen_ids is not None:
        seen_ids.add(record_id)

    return {
        "id": record_id,
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
        "source_id": record_id,
    }
