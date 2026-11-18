import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from clickless.config import settings
from clickless.data import load_catalog
from clickless.data.pipeline import _subsystem
from clickless.data.repository import validate_catalog
from clickless.retail import BestBuyClient, EbayClient
from clickless.sources import AutoDevClient, MarketCheckClient, SerpApiClient
from clickless.sources.base import ProductOffer, SourceError, VehicleListing


def _slug(value: object) -> str:
    return "-".join("".join(character if character.isalnum() else " " for character in str(value).casefold()).split())


def _add_listing(catalog: dict, vehicle: dict, item: VehicleListing) -> bool:
    record_id = f"{_slug(item.source)}-listing-{_slug(item.identifier)}"
    if any(record["id"] == record_id for record in catalog["listings"]):
        return False
    catalog["listings"].append(
        {
            "id": record_id,
            "vehicle_id": vehicle["id"],
            "price": item.price,
            "mileage": item.mileage,
            "location": item.location,
            "vin": item.vin,
            "source_id": record_id,
            "source_name": item.source,
            "source_url": item.url,
            "captured_at": item.captured_at,
            "data_mode": "live",
        }
    )
    return True


def refresh_live_inventory(
    catalog: dict,
    per_vehicle: int,
    max_vehicles: int | None = None,
) -> tuple[int, list[str]]:
    marketcheck = MarketCheckClient.from_env()
    auto_dev = AutoDevClient.from_env()
    enabled_sources = {
        name
        for name, client in (("MarketCheck", marketcheck), ("Auto.dev", auto_dev))
        if client is not None
    }
    target_vehicles = catalog["vehicles"][:max_vehicles]
    target_ids = {vehicle["id"] for vehicle in target_vehicles}
    if enabled_sources:
        catalog["listings"] = [
            item
            for item in catalog["listings"]
            if not (
                item.get("vehicle_id") in target_ids
                and item.get("data_mode") == "live"
                and item.get("source_name") in enabled_sources
            )
        ]
    added = 0
    used_sources: list[str] = []
    for vehicle in target_vehicles:
        results: list[VehicleListing] = []
        if marketcheck is not None:
            try:
                results = marketcheck.search(
                    make=vehicle["make"],
                    model=vehicle["model"],
                    year=vehicle["year"],
                    limit=per_vehicle,
                )
                if results:
                    used_sources.append("MarketCheck")
            except SourceError as error:
                print(f"MarketCheck {vehicle['id']}: {error}")
        if not results and auto_dev is not None:
            try:
                results = auto_dev.search(
                    make=vehicle["make"],
                    model=vehicle["model"],
                    year=vehicle["year"],
                    limit=per_vehicle,
                )
                if results:
                    used_sources.append("Auto.dev")
            except SourceError as error:
                print(f"Auto.dev {vehicle['id']}: {error}")
        for item in results[:per_vehicle]:
            added += int(_add_listing(catalog, vehicle, item))
    return added, list(dict.fromkeys(used_sources))


def _add_product(catalog: dict, vehicle: dict, item: ProductOffer) -> bool:
    record_id = f"serpapi-product-{_slug(item.identifier)}-{vehicle['id']}"
    if any(record["id"] == record_id for record in catalog["parts"]):
        return False
    catalog["parts"].append(
        {
            "id": record_id,
            "vehicle_id": vehicle["id"],
            "name": item.title,
            "price": item.price,
            "subsystem": _subsystem(item.title),
            "verified_fitment": False,
            "rating": item.rating,
            "review_count": item.review_count,
            "review_snippets": list(item.review_texts),
            "merchant": item.merchant,
            "source_id": record_id,
            "source_name": item.source,
            "source_url": item.url,
            "captured_at": item.captured_at,
            "data_mode": "live",
        }
    )
    return True


def refresh_serpapi(
    catalog: dict,
    per_vehicle: int,
    max_vehicles: int | None = None,
) -> int:
    client = SerpApiClient.from_env()
    if client is None:
        return 0
    target_vehicles = catalog["vehicles"][:max_vehicles]
    target_ids = {vehicle["id"] for vehicle in target_vehicles}
    catalog["parts"] = [
        item
        for item in catalog["parts"]
        if not (
            item.get("data_mode") == "live"
            and item.get("source_name") == "SerpApi Google Shopping"
            and item.get("vehicle_id") in target_ids
        )
    ]
    added = 0
    for vehicle in target_vehicles:
        query = f"{vehicle['year']} {vehicle['make']} {vehicle['model']} brake pads air filter battery"
        try:
            offers = client.search_products(query, limit=per_vehicle)
        except SourceError as error:
            print(f"SerpApi {vehicle['id']}: {error}")
            continue
        for offer in offers:
            added += int(_add_product(catalog, vehicle, offer))
    return added


def refresh_ebay(catalog: dict, per_vehicle: int) -> int:
    client = EbayClient.from_env()
    if client is None:
        return 0
    added = 0
    existing = {item["id"] for item in catalog["listings"]}
    captured = datetime.now(timezone.utc).isoformat()
    for vehicle in catalog["vehicles"]:
        query = f"{vehicle['year']} {vehicle['make']} {vehicle['model']}"
        for item in client.search(query, limit=per_vehicle * 3):
            mileage = item.attributes.get("mileage")
            if item.price is None or mileage is None:
                continue
            record_id = f"ebay-listing-{item.identifier}"
            if record_id in existing:
                continue
            catalog["listings"].append(
                {
                    "id": record_id,
                    "vehicle_id": vehicle["id"],
                    "price": int(item.price),
                    "mileage": int(mileage),
                    "location": str(item.attributes.get("location") or ""),
                    "vin": "",
                    "source_id": record_id,
                    "source_name": "eBay Browse API",
                    "source_url": item.url,
                    "captured_at": captured,
                    "data_mode": "live",
                }
            )
            existing.add(record_id)
            added += 1
            if added and sum(
                listing["vehicle_id"] == vehicle["id"] and listing.get("data_mode") == "live"
                for listing in catalog["listings"]
            ) >= per_vehicle:
                break
    return added


def refresh_bestbuy(catalog: dict, per_vehicle: int) -> int:
    client = BestBuyClient.from_env()
    if client is None:
        return 0
    added = 0
    existing = {item["id"] for item in catalog["parts"]}
    captured = datetime.now(timezone.utc).isoformat()
    for vehicle in catalog["vehicles"]:
        query = f"{vehicle['make']} {vehicle['model']} car electronics"
        for item in client.search(query, limit=per_vehicle):
            if item.price is None:
                continue
            record_id = f"bestbuy-product-{item.identifier}-{vehicle['id']}"
            if record_id in existing:
                continue
            catalog["parts"].append(
                {
                    "id": record_id,
                    "vehicle_id": vehicle["id"],
                    "name": item.title,
                    "price": item.price,
                    "subsystem": _subsystem(item.title),
                    "verified_fitment": False,
                    "rating": item.attributes.get("rating"),
                    "source_id": record_id,
                    "source_name": "Best Buy Products API",
                    "source_url": item.url,
                    "captured_at": captured,
                    "data_mode": "live",
                }
            )
            existing.add(record_id)
            added += 1
    return added


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-vehicle", type=int, default=5)
    parser.add_argument("--max-vehicles", type=int, default=0)
    parser.add_argument("--output", default=settings.catalog_path)
    args = parser.parse_args()
    catalog = load_catalog()
    max_vehicles = args.max_vehicles or None
    inventory, inventory_sources = refresh_live_inventory(
        catalog, args.per_vehicle, max_vehicles
    )
    serpapi = refresh_serpapi(catalog, args.per_vehicle, max_vehicles)
    ebay = refresh_ebay(catalog, args.per_vehicle)
    bestbuy = refresh_bestbuy(catalog, args.per_vehicle)
    live_source_names = {
        item["source_name"]
        for collection in (catalog["listings"], catalog["parts"])
        for item in collection
        if item.get("data_mode") == "live"
    }
    catalog["snapshot"]["sources"] = [
        source
        for source in catalog["snapshot"]["sources"]
        if source not in {"MarketCheck", "Auto.dev", "SerpApi Google Shopping"}
    ]
    catalog["snapshot"]["sources"].extend(sorted(live_source_names))
    if ebay:
        catalog["snapshot"]["sources"].append("eBay Browse API")
    if bestbuy:
        catalog["snapshot"]["sources"].append("Best Buy Products API")
    catalog["snapshot"]["sources"] = list(dict.fromkeys(catalog["snapshot"]["sources"]))
    validate_catalog(catalog)
    output = Path(args.output)
    output.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    print(
        f"Added {inventory} live vehicle listings, {serpapi} SerpApi products, "
        f"{ebay} eBay listings, and {bestbuy} Best Buy products."
    )


if __name__ == "__main__":
    main()
