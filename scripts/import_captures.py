"""Import Playwright JSON-LD captures (clickless-crawl) into the catalog as live_capture records.

Usage:
    clickless-crawl walmart "2009 Toyota RAV4 battery"
    python scripts/import_captures.py
"""
import argparse
import json
from pathlib import Path

from clickless.config import settings
from clickless.data import load_catalog
from clickless.data.pipeline import _subsystem
from clickless.data.repository import validate_catalog


def _vehicle_for(query: str, vehicles: list[dict]) -> dict | None:
    text = query.casefold()
    for vehicle in vehicles:
        if (
            str(vehicle["year"]) in text
            and vehicle["make"].casefold() in text
            and vehicle["model"].casefold() in text
        ):
            return vehicle
    return None


def import_captures(catalog: dict, capture_dir: Path) -> int:
    existing = {item["id"] for item in catalog["parts"]}
    added = 0
    for path in sorted(capture_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        vehicle = _vehicle_for(str(payload.get("query", "")), catalog["vehicles"])
        if vehicle is None:
            print(f"skip {path.name}: query does not name a catalog vehicle")
            continue
        for index, product in enumerate(payload.get("products", [])):
            if product.get("price") is None or not product.get("name") or not product.get("url"):
                continue
            record_id = f"capture-{payload['site']}-{path.stem}-{index}"
            if record_id in existing:
                continue
            catalog["parts"].append(
                {
                    "id": record_id,
                    "vehicle_id": vehicle["id"],
                    "name": product["name"],
                    "price": float(product["price"]),
                    "subsystem": _subsystem(product["name"]),
                    "verified_fitment": False,
                    "rating": product.get("rating"),
                    "source_id": record_id,
                    "source_name": f"{payload['site'].title()} (Playwright capture)",
                    "source_url": product["url"],
                    "captured_at": payload["captured_at"],
                    "selector_version": payload.get("selector_version", "jsonld-v1"),
                    "data_mode": "live_capture",
                }
            )
            existing.add(record_id)
            added += 1
    return added


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--captures", default=settings.root / "data" / "cache" / "retail")
    parser.add_argument("--output", default=settings.catalog_path)
    args = parser.parse_args()
    catalog = load_catalog()
    added = import_captures(catalog, Path(args.captures))
    validate_catalog(catalog)
    Path(args.output).write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    print(f"Imported {added} captured products.")


if __name__ == "__main__":
    main()
