from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from clickless.config import settings

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CATALOG = settings.catalog_path
EVIDENCE_COLLECTIONS = ("listings", "safety_ratings", "recalls", "reviews", "parts")
PROVENANCE_FIELDS = ("source_id", "source_name", "source_url", "captured_at")


class CatalogError(ValueError):
    pass


def load_catalog(path: str | Path | None = None) -> dict[str, Any]:
    configured = os.getenv("CLICKLESS_CATALOG_PATH")
    catalog_path = Path(path or configured or DEFAULT_CATALOG)
    if not catalog_path.is_absolute():
        catalog_path = ROOT / catalog_path
    if not catalog_path.exists():
        raise CatalogError(
            f"Processed catalog not found at {catalog_path}. Run clickless-build-data first."
        )
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    validate_catalog(catalog)
    return catalog


def validate_catalog(catalog: dict[str, Any]) -> None:
    snapshot = catalog.get("snapshot")
    if not isinstance(snapshot, dict) or not snapshot.get("id"):
        raise CatalogError("Catalog snapshot is missing an id")

    vehicles = catalog.get("vehicles")
    if not isinstance(vehicles, list) or not vehicles:
        raise CatalogError("Catalog must contain vehicles")

    vehicle_ids = _unique_ids(vehicles, "vehicles")
    aspect_ids: set[str] = set()

    for collection_name in EVIDENCE_COLLECTIONS:
        records = catalog.get(collection_name)
        if not isinstance(records, list):
            raise CatalogError(f"Catalog collection {collection_name} must be a list")
        _unique_ids(records, collection_name)
        for record in records:
            if record.get("vehicle_id") not in vehicle_ids:
                raise CatalogError(f"Unknown vehicle_id in {collection_name}: {record.get('vehicle_id')}")
            missing = [field for field in PROVENANCE_FIELDS if not record.get(field)]
            if missing:
                raise CatalogError(f"Missing provenance in {collection_name}/{record.get('id')}: {missing}")
            for aspect in record.get("aspects", []):
                aspect_id = aspect.get("id")
                if not aspect_id or aspect_id in aspect_ids:
                    raise CatalogError(f"Invalid or duplicate aspect id: {aspect_id}")
                aspect_ids.add(aspect_id)


def vehicle_bundle(catalog: dict[str, Any], vehicle_id: str) -> dict[str, Any]:
    vehicle = next((item for item in catalog["vehicles"] if item["id"] == vehicle_id), None)
    if vehicle is None:
        raise CatalogError(f"Unknown vehicle: {vehicle_id}")

    bundle: dict[str, Any] = {"vehicle": vehicle}
    for collection_name in EVIDENCE_COLLECTIONS:
        bundle[collection_name] = [
            item for item in catalog[collection_name] if item["vehicle_id"] == vehicle_id
        ]
    return bundle


def all_vehicle_bundles(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    return [vehicle_bundle(catalog, vehicle["id"]) for vehicle in catalog["vehicles"]]


def _unique_ids(records: list[dict[str, Any]], collection_name: str) -> set[str]:
    ids = [record.get("id") for record in records]
    if any(not record_id for record_id in ids):
        raise CatalogError(f"Every {collection_name} record needs an id")
    if len(ids) != len(set(ids)):
        raise CatalogError(f"Duplicate ids in {collection_name}")
    return set(ids)
