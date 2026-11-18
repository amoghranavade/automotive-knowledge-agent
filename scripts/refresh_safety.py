"""Re-fetch NHTSA crash ratings and recalls for the catalog without rebuilding everything.

Fixes "Safety: No data" for 2007-2010 cars (NHTSA has no overall rating, only
per-test stars) and model names that carry a trim ("cr-v ex-l").

    python scripts/refresh_safety.py            # update data/processed/catalog.json
    python scripts/refresh_safety.py --report   # only print what is missing
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from clickless.config import settings
from clickless.data import load_catalog
from clickless.data.nhtsa import NHTSAError, fetch_recalls, fetch_safety_rating, safety_stars
from clickless.data.repository import validate_catalog


def report(catalog: dict) -> None:
    ratings = {item["vehicle_id"]: item for item in catalog["safety_ratings"]}
    recalls: dict[str, int] = {}
    for item in catalog["recalls"]:
        recalls[item["vehicle_id"]] = recalls.get(item["vehicle_id"], 0) + 1
    print(f"{'vehicle':40} {'model string':22} {'safety':>8} {'basis':>13} {'recalls':>7}")
    for vehicle in catalog["vehicles"]:
        stars, basis = safety_stars(ratings.get(vehicle["id"]))
        print(
            f"{vehicle['id'][:40]:40} {vehicle['model'][:22]:22} "
            f"{str(stars if stars is not None else '-'):>8} {basis:>13} {recalls.get(vehicle['id'], 0):>7}"
        )


def refresh(catalog: dict) -> tuple[int, int]:
    captured = datetime.now(timezone.utc).isoformat()
    ratings, recalls = [], []
    for vehicle in catalog["vehicles"]:
        try:
            rating = fetch_safety_rating(
                vehicle["make"], vehicle["model"], vehicle["year"], drivetrain=vehicle["drivetrain"]
            )
        except NHTSAError as error:
            print(f"rating {vehicle['id']}: {error}")
            rating = None
        if rating:
            source_id = f"nhtsa-rating-{vehicle['id']}-{rating['vehicle_id']}"
            ratings.append(
                {
                    "id": source_id,
                    "vehicle_id": vehicle["id"],
                    **{key: value for key, value in rating.items() if key != "vehicle_id"},
                    "source_id": source_id,
                    "source_name": "NHTSA NCAP",
                }
            )
        try:
            found = fetch_recalls(vehicle["make"], vehicle["model"], vehicle["year"])
        except NHTSAError as error:
            print(f"recalls {vehicle['id']}: {error}")
            found = [item for item in catalog["recalls"] if item["vehicle_id"] == vehicle["id"]]
            recalls.extend(found)
            continue
        seen = set()
        for item in found:
            item["vehicle_id"] = vehicle["id"]
            item["id"] = f"{vehicle['id']}-{item['id']}"
            if item["id"] in seen:
                continue
            seen.add(item["id"])
            item["captured_at"] = item.get("captured_at") or captured
            recalls.append(item)
    catalog["safety_ratings"] = ratings
    catalog["recalls"] = recalls
    return len(ratings), len(recalls)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--output", default=settings.catalog_path)
    args = parser.parse_args()
    catalog = load_catalog()
    if args.report:
        report(catalog)
        return
    counts = refresh(catalog)
    validate_catalog(catalog)
    Path(args.output).write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    print(f"Saved {counts[0]} crash-rating records and {counts[1]} recalls.")
    report(catalog)


if __name__ == "__main__":
    main()
