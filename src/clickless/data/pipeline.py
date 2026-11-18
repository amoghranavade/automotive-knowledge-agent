from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import re
import shutil
import subprocess
import sys
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests

from clickless.config import settings
from clickless.data.aspects import extract_aspects, normalize_subsystem
from clickless.data.nhtsa import NHTSAError, fetch_recalls, fetch_safety_rating
from clickless.data.repository import validate_catalog


KAGGLE_DATASET = "austinreese/craigslist-carstrucks-data"
KAGGLE_DOWNLOAD_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/austinreese/craigslist-carstrucks-data"
)
OPINRANK_URL = "https://archive.ics.uci.edu/static/public/205/opinrank+review+dataset.zip"
AMAZON_REVIEW_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Automotive.jsonl.gz"
AMAZON_META_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Automotive.jsonl.gz"
BODY_TYPES = {
    "suv": "SUV",
    "sedan": "Sedan",
    "pickup": "Truck",
    "truck": "Truck",
}
DRIVETRAINS = {"4wd": "4WD", "fwd": "FWD", "rwd": "RWD", "awd": "AWD"}
AWD_CROSSOVERS = {
    ("subaru", "forester"),
    ("subaru", "outback"),
    ("subaru", "impreza"),
    ("subaru", "legacy"),
    ("toyota", "rav4"),
    ("toyota", "highlander"),
    ("nissan", "murano"),
    ("nissan", "rogue"),
    ("honda", "cr-v"),
    ("honda", "pilot"),
    ("mazda", "cx-7"),
    ("mazda", "cx-9"),
    ("lexus", "rx 350"),
    ("ford", "escape"),
    ("ford", "edge"),
}
_NOISY_MODEL_TOKENS = {
    "1owner",
    "one",
    "owner",
    "service",
    "clean",
    "title",
    "only",
    "miles",
    "low",
    "cab",
    "crew",
    "supercab",
    "4x4",
    "awd",
    "4wd",
    "fwd",
    "rwd",
    "auto",
    "automatic",
    "manual",
    "v6",
    "v8",
}
_MODEL_NORMALIZATIONS = {
    "f150": "F-150",
    "f-150": "F-150",
    "f250": "F-250",
    "f-250": "F-250",
    "f350": "F-350",
    "f-350": "F-350",
    "rav4": "RAV4",
    "cr-v": "CR-V",
    "crv": "CR-V",
    "fj cruiser": "FJ Cruiser",
    "es 350": "ES 350",
    "es350": "ES 350",
    "rx 350": "RX 350",
    "rx350": "RX 350",
}
_JUNK_MODELS = {"dr", "truck", "sedan", "suv", "car"}
GENERATION_RANGES = {
    ("toyota", "fj cruiser"): ((2007, 2014, 2007),),
    ("honda", "accord"): ((2003, 2007, 2003), (2008, 2012, 2008)),
    ("jeep", "wrangler"): ((2007, 2018, 2007),),
    ("toyota", "camry"): ((2007, 2011, 2007),),
    ("ford", "f-150"): ((2004, 2008, 2004), (2009, 2014, 2009)),
    ("toyota", "corolla"): ((2003, 2008, 2003), (2009, 2013, 2009)),
    ("lexus", "es 350"): ((2007, 2012, 2007),),
}


def _slug(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", str(value or "").casefold()).strip("-")
    return text or "unknown"


def _number(value: Any, kind: type[int] | type[float]) -> int | float | None:
    try:
        result = kind(float(str(value).replace(",", "")))
    except (TypeError, ValueError):
        return None
    return result


def _model_name(make: str, value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value).strip()
    cleaned = re.sub(rf"^{re.escape(make)}\s+", "", cleaned, flags=re.I)
    tokens: list[str] = []
    for raw_token in cleaned.split():
        token = raw_token.strip(" -,/")
        if not token:
            continue
        if token.casefold() in _NOISY_MODEL_TOKENS:
            if tokens:
                break
            continue
        tokens.append(token)
        if len(tokens) == 3:
            break
    if not tokens:
        return ""
    joined = " ".join(tokens).strip(" -")
    normalized = _MODEL_NORMALIZATIONS.get(joined.casefold())
    if normalized is None:
        formatted = []
        for token in tokens:
            mapped = _MODEL_NORMALIZATIONS.get(token.casefold())
            if mapped:
                formatted.append(mapped)
            elif token.isdigit():
                formatted.append(token)
            elif any(ch.isalpha() for ch in token):
                formatted.append(token.title())
            else:
                formatted.append(token.upper())
        normalized = " ".join(formatted).strip(" -")
        normalized = _MODEL_NORMALIZATIONS.get(normalized.casefold(), normalized)
    if len(normalized) < 2 or normalized.casefold() in _JUNK_MODELS:
        return ""
    return normalized


def generation_id(make: str, model: str, year: int) -> str:
    start = year
    for lower, upper, generation_start in GENERATION_RANGES.get(
        (make.casefold(), model.casefold()), ()
    ):
        if lower <= year <= upper:
            start = generation_start
            break
    return f"{_slug(make)}-{_slug(model)}-generation-{start}"


def craigslist_catalog(
    path: Path,
    max_models: int = 30,
    max_listings_per_model: int = 40,
) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            year = _number(row.get("year"), int)
            price = _number(row.get("price"), int)
            mileage = _number(row.get("odometer"), int)
            make = str(row.get("manufacturer") or "").strip().title()
            model = _model_name(make, str(row.get("model") or ""))
            body_type = BODY_TYPES.get(str(row.get("type") or "").casefold())
            drivetrain = DRIVETRAINS.get(str(row.get("drive") or "").casefold())
            if drivetrain == "4WD" and (make.casefold(), model.casefold()) in AWD_CROSSOVERS:
                drivetrain = "AWD"
            if not all((year, price, mileage, make, model, body_type, drivetrain)):
                continue
            if not (2000 <= year <= 2027 and 2_000 <= price <= 80_000 and 500 <= mileage <= 250_000):
                continue
            key = (year, make, model, body_type, drivetrain)
            records = groups[key]
            if len(records) < max_listings_per_model:
                records.append(row)

    eligible = [(key, rows) for key, rows in groups.items() if len(rows) >= 5]
    eligible.sort(key=lambda item: (-len(item[1]), item[0]))
    review_compatible = [item for item in eligible if item[0][0] in {2007, 2008, 2009}]
    selected = _balanced_groups(review_compatible, max_models)
    if len(selected) < max_models:
        selected_keys = {item[0] for item in selected}
        selected.extend(
            item
            for item in _balanced_groups(eligible, max_models)
            if item[0] not in selected_keys
        )
        selected = selected[:max_models]
    if len(selected) < min(20, max_models):
        raise ValueError(
            f"Only {len(selected)} vehicle groups have five usable listings; at least 20 are required."
        )

    captured = datetime.now(timezone.utc).isoformat()
    vehicles = []
    listings = []
    for (year, make, model, body_type, drivetrain), rows in selected:
        vehicle_id = f"{_slug(make)}-{_slug(model)}-{year}-{drivetrain.casefold()}"
        vehicles.append(
            {
                "id": vehicle_id,
                "generation_id": generation_id(make, model, year),
                "year": year,
                "make": make,
                "model": model,
                "trim": "",
                "body_type": body_type,
                "drivetrain": drivetrain,
            }
        )
        for index, row in enumerate(rows):
            listing_id = str(row.get("id") or f"{vehicle_id}-{index}")
            source_url = str(row.get("url") or "").strip() or "https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data"
            listings.append(
                {
                    "id": f"listing-{listing_id}",
                    "vehicle_id": vehicle_id,
                    "price": int(float(row["price"])),
                    "mileage": int(float(row["odometer"])),
                    "location": ", ".join(filter(None, [row.get("region"), row.get("state")])),
                    "vin": str(row.get("VIN") or "").strip(),
                    "source_id": f"craigslist-{listing_id}",
                    "source_name": "Craigslist snapshot via Kaggle",
                    "source_url": source_url,
                    "captured_at": str(row.get("posting_date") or captured),
                    "data_mode": "frozen_snapshot",
                }
            )

    return {
        "snapshot": {
            "id": f"clickless-{datetime.now(timezone.utc).date().isoformat()}",
            "created_at": captured,
            "mode": "frozen_snapshot",
            "notice": "Recommendations use research datasets and optional live sources. Listing snapshots may no longer be active; verify all details before purchase.",
            "sources": ["Kaggle Used Cars Dataset"],
        },
        "vehicles": vehicles,
        "listings": listings,
        "safety_ratings": [],
        "recalls": [],
        "reviews": [],
        "parts": [],
    }


def _balanced_groups(
    groups: list[tuple[tuple[Any, ...], list[dict[str, Any]]]],
    limit: int,
) -> list[tuple[tuple[Any, ...], list[dict[str, Any]]]]:
    buckets: dict[str, list[tuple[tuple[Any, ...], list[dict[str, Any]]]]] = defaultdict(list)
    for item in groups:
        buckets[str(item[0][3])].append(item)
    selected = []
    while len(selected) < limit and any(buckets.values()):
        for body in ("SUV", "Sedan", "Truck"):
            if buckets[body] and len(selected) < limit:
                selected.append(buckets[body].pop(0))
    return selected


def add_opinrank_reviews(
    catalog: dict[str, Any],
    archive: Path,
    max_reviews: int = 2_000,
    per_vehicle: int = 150,
) -> int:
    targets = {
        vehicle["id"]: {
            _slug(vehicle["make"]),
            _slug(vehicle["model"]),
            str(vehicle["year"]),
        }
        for vehicle in catalog["vehicles"]
    }
    counts: dict[str, int] = defaultdict(int)
    retained = 0
    with zipfile.ZipFile(archive) as outer:
        nested_name = next(
            (name for name in outer.namelist() if name.casefold().endswith("opinrankdataset.zip")),
            None,
        )
        nested = zipfile.ZipFile(io.BytesIO(outer.read(nested_name))) if nested_name else None
        bundle = nested or outer
        try:
            retained = _read_opinrank_bundle(
                bundle, catalog, targets, counts, retained, max_reviews, per_vehicle
            )
        finally:
            if nested is not None:
                nested.close()
    if retained:
        catalog["snapshot"]["sources"].append("UCI OpinRank")
    return retained


def _read_opinrank_bundle(
    bundle: zipfile.ZipFile,
    catalog: dict[str, Any],
    targets: dict[str, set[str]],
    counts: dict[str, int],
    retained: int,
    max_reviews: int,
    per_vehicle: int,
) -> int:
    for name in bundle.namelist():
        if retained >= max_reviews or name.endswith("/") or "car" not in name.casefold():
            continue
        normalized_path = _slug(name)
        vehicle_id = next(
            (
                item_id for item_id, tokens in targets.items()
                if all(token in normalized_path for token in tokens)
                and counts[item_id] < per_vehicle
            ),
            None,
        )
        if vehicle_id is None:
            continue
        raw = bundle.read(name).decode("utf-8", errors="replace")
        for text in _review_blocks(raw):
            if retained >= max_reviews or counts[vehicle_id] >= per_vehicle:
                break
            review_id = f"opinrank-{_slug(name)}-{counts[vehicle_id]}"
            vehicle = next(item for item in catalog["vehicles"] if item["id"] == vehicle_id)
            catalog["reviews"].append(
                {
                    "id": review_id,
                    "vehicle_id": vehicle_id,
                    "review_year": vehicle["year"],
                    "text": text,
                    "rating": None,
                    "match_type": "exact_year",
                    "source_kind": "owner_review",
                    "aspects": extract_aspects(text, review_id),
                    "source_id": review_id,
                    "source_name": "Edmunds OpinRank",
                    "source_url": "https://archive.ics.uci.edu/dataset/205/opinrank+review+dataset",
                    "captured_at": catalog["snapshot"]["created_at"],
                    "data_mode": "frozen_snapshot",
                }
            )
            counts[vehicle_id] += 1
            retained += 1
    return retained


def _review_blocks(raw: str) -> Iterable[str]:
    documents = re.findall(r"<DOC>(.*?)</DOC>", raw, flags=re.I | re.S)
    if documents:
        for document in documents:
            text_match = re.search(r"<TEXT>(.*?)</TEXT>", document, flags=re.I | re.S)
            favorite_match = re.search(r"<FAVORITE>(.*?)</FAVORITE>", document, flags=re.I | re.S)
            fields = [
                match.group(1).strip()
                for match in (text_match, favorite_match)
                if match and match.group(1).strip()
            ]
            cleaned = re.sub(r"\s+", " ", " ".join(fields)).strip()
            if 80 <= len(cleaned) <= 8_000:
                yield cleaned
        return
    text = re.sub(r"<[^>]+>", " ", raw)
    text = re.sub(r"\r", "", text)
    blocks = re.split(r"\n\s*\n|(?=\b(?:Review|AUTHOR|DATE)\b)", text)
    for block in blocks:
        cleaned = re.sub(r"\s+", " ", block).strip()
        if 80 <= len(cleaned) <= 8_000:
            yield cleaned


def add_amazon_products(
    catalog: dict[str, Any],
    metadata_path: Path,
    reviews_path: Path,
    max_products: int = 300,
) -> int:
    vehicles = catalog["vehicles"]
    products: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for record in _json_lines(metadata_path):
        title = str(record.get("title") or "")
        lowered = title.casefold()
        vehicle = next(
            (
                item for item in vehicles
                if item["make"].casefold() in lowered and item["model"].casefold() in lowered
            ),
            None,
        )
        price = _number(record.get("price"), float)
        if vehicle is None or price is None:
            continue
        parent_asin = str(record.get("parent_asin") or "")
        if not parent_asin:
            continue
        products[parent_asin] = (vehicle, record)
        if len(products) >= max_products:
            break

    review_by_asin: dict[str, dict[str, Any]] = {}
    for review in _json_lines(reviews_path):
        parent_asin = str(review.get("parent_asin") or "")
        if parent_asin in products and parent_asin not in review_by_asin:
            review_by_asin[parent_asin] = review
        if len(review_by_asin) == len(products):
            break

    for parent_asin, (vehicle, product) in products.items():
        title = str(product.get("title") or "Automotive product")
        subsystem = _subsystem(title)
        source_url = f"https://www.amazon.com/dp/{parent_asin}"
        catalog["parts"].append(
            {
                "id": f"amazon-product-{parent_asin}",
                "vehicle_id": vehicle["id"],
                "name": title,
                "price": float(product["price"]),
                "subsystem": subsystem,
                "verified_fitment": False,
                "rating": product.get("average_rating"),
                "source_id": f"amazon-meta-{parent_asin}",
                "source_name": "Amazon Reviews 2023 metadata",
                "source_url": source_url,
                "captured_at": catalog["snapshot"]["created_at"],
                "data_mode": "frozen_snapshot",
            }
        )
        review = review_by_asin.get(parent_asin)
        if review and str(review.get("text") or "").strip():
            review_id = f"amazon-review-{review.get('asin') or parent_asin}"
            text = str(review["text"]).strip()
            catalog["reviews"].append(
                {
                    "id": review_id,
                    "vehicle_id": vehicle["id"],
                    "review_year": vehicle["year"],
                    "text": text,
                    "rating": review.get("rating"),
                    "match_type": "product_fitment_unverified",
                    "source_kind": "product_review",
                    "aspects": extract_aspects(text, review_id),
                    "source_id": review_id,
                    "source_name": "Amazon Reviews 2023",
                    "source_url": source_url,
                    "captured_at": catalog["snapshot"]["created_at"],
                    "data_mode": "frozen_snapshot",
                }
            )
    if products:
        catalog["snapshot"]["sources"].append("Amazon Reviews 2023 Automotive")
    return len(products)


def _json_lines(path: Path) -> Iterable[dict[str, Any]]:
    handle = (
        gzip.open(path, "rt", encoding="utf-8", errors="replace")
        if path.suffix == ".gz"
        else path.open("r", encoding="utf-8", errors="replace")
    )
    with handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                yield value


def _subsystem(text: str) -> str:
    return normalize_subsystem(text)


def add_nhtsa(catalog: dict[str, Any]) -> tuple[int, int]:
    rating_count = 0
    recall_count = 0
    captured = datetime.now(timezone.utc).isoformat()
    for vehicle in catalog["vehicles"]:
        try:
            rating = fetch_safety_rating(vehicle["make"], vehicle["model"], vehicle["year"])
        except NHTSAError:
            rating = None
        if rating:
            source_id = f"nhtsa-rating-{vehicle['id']}-{rating['vehicle_id']}"
            catalog["safety_ratings"].append(
                {
                    "id": source_id,
                    "vehicle_id": vehicle["id"],
                    **{key: value for key, value in rating.items() if key != "vehicle_id"},
                    "source_id": source_id,
                    "source_name": "NHTSA NCAP",
                }
            )
            rating_count += 1
        try:
            recalls = fetch_recalls(vehicle["make"], vehicle["model"], vehicle["year"])
        except NHTSAError:
            recalls = []
        seen_recall_ids = {item["id"] for item in catalog["recalls"]}
        for idx, item in enumerate(recalls):
            item["vehicle_id"] = vehicle["id"]
            base_id = f"{vehicle['id']}-{item['id']}"
            recall_id = base_id
            suffix = idx
            while recall_id in seen_recall_ids:
                recall_id = f"{base_id}-{suffix}"
                suffix += 1
            seen_recall_ids.add(recall_id)
            item["id"] = recall_id
            item["captured_at"] = item.get("captured_at") or captured
            catalog["recalls"].append(item)
            recall_count += 1
    if rating_count or recall_count:
        catalog["snapshot"]["sources"].append("NHTSA")
    return rating_count, recall_count


def download_public_data(downloads: Path, amazon_records: int = 50_000) -> dict[str, Path]:
    downloads.mkdir(parents=True, exist_ok=True)
    craigslist_dir = downloads / "craigslist"
    craigslist_dir.mkdir(exist_ok=True)
    vehicles_csv = craigslist_dir / "vehicles.csv"
    if not vehicles_csv.exists():
        downloaded = False
        try:
            import kagglehub

            dataset_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))
            candidate_csv = next(dataset_dir.rglob("vehicles.csv"), None)
            if candidate_csv is not None and candidate_csv.is_file():
                shutil.copy2(candidate_csv, vehicles_csv)
                downloaded = vehicles_csv.exists()
        except Exception:
            downloaded = False

        if not downloaded:
            venv_kaggle = Path(sys.executable).with_name(
                "kaggle.exe" if sys.platform.startswith("win") else "kaggle"
            )
            kaggle_bin = str(venv_kaggle) if venv_kaggle.exists() else shutil.which("kaggle")
            if kaggle_bin:
                try:
                    subprocess.run(
                        [kaggle_bin, "datasets", "download", KAGGLE_DATASET, "--path", str(craigslist_dir), "--unzip"],
                        check=True,
                    )
                    downloaded = vehicles_csv.exists()
                except (subprocess.SubprocessError, OSError):
                    downloaded = False

        if not downloaded:
            archive_path = craigslist_dir / "craigslist-carstrucks-data.zip"
            try:
                _download(KAGGLE_DOWNLOAD_URL, archive_path)
                with zipfile.ZipFile(archive_path) as bundle:
                    csv_member = next(
                        (
                            name
                            for name in bundle.namelist()
                            if Path(name).name.casefold() == "vehicles.csv"
                        ),
                        None,
                    )
                    if csv_member is None:
                        raise RuntimeError("Kaggle archive did not contain vehicles.csv")
                    if csv_member == "vehicles.csv":
                        bundle.extract(csv_member, craigslist_dir)
                    else:
                        with bundle.open(csv_member) as source, vehicles_csv.open("wb") as target:
                            shutil.copyfileobj(source, target)
            finally:
                archive_path.unlink(missing_ok=True)

    opinrank = downloads / "OpinRankDataset.zip"
    if not opinrank.exists():
        _download(OPINRANK_URL, opinrank)

    amazon_dir = downloads / "amazon"
    amazon_dir.mkdir(exist_ok=True)
    amazon_reviews = amazon_dir / "Automotive.sample.jsonl"
    amazon_meta = amazon_dir / "meta_Automotive.sample.jsonl"
    if not amazon_reviews.exists():
        _stream_gzip_json(AMAZON_REVIEW_URL, amazon_reviews, amazon_records)
    if not amazon_meta.exists():
        _stream_gzip_json(AMAZON_META_URL, amazon_meta, amazon_records)
    return {
        "craigslist": vehicles_csv,
        "opinrank": opinrank,
        "amazon_reviews": amazon_reviews,
        "amazon_meta": amazon_meta,
    }


def _download(url: str, destination: Path) -> None:
    with requests.get(url, stream=True, timeout=(10, 120)) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)


def _stream_gzip_json(url: str, destination: Path, records: int) -> None:
    with requests.get(url, stream=True, timeout=(10, 120)) as response:
        response.raise_for_status()
        response.raw.decode_content = False
        with gzip.GzipFile(fileobj=response.raw) as source, destination.open("wb") as output:
            for index, line in enumerate(source):
                if index >= records:
                    break
                output.write(line)


def build_catalog(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "craigslist": Path(args.craigslist),
        "opinrank": Path(args.opinrank),
        "amazon_reviews": Path(args.amazon_reviews),
        "amazon_meta": Path(args.amazon_meta),
    }
    if args.download_public:
        paths.update(download_public_data(settings.downloads_path, args.amazon_records))
    if not paths["craigslist"].exists():
        raise FileNotFoundError(f"Craigslist CSV not found: {paths['craigslist']}")

    catalog = craigslist_catalog(paths["craigslist"], args.max_models, args.listings_per_model)
    if paths["opinrank"].exists():
        add_opinrank_reviews(catalog, paths["opinrank"], args.max_reviews)
    if paths["amazon_meta"].exists() and paths["amazon_reviews"].exists():
        add_amazon_products(catalog, paths["amazon_meta"], paths["amazon_reviews"])
    if not args.skip_nhtsa:
        add_nhtsa(catalog)
    validate_catalog(catalog)
    return catalog


def main() -> None:
    defaults = settings.downloads_path
    parser = argparse.ArgumentParser(description="Build the ClickLess research catalog")
    parser.add_argument("--download-public", action="store_true")
    parser.add_argument("--craigslist", default=defaults / "craigslist" / "vehicles.csv")
    parser.add_argument("--opinrank", default=defaults / "OpinRankDataset.zip")
    parser.add_argument("--amazon-reviews", default=defaults / "amazon" / "Automotive.sample.jsonl")
    parser.add_argument("--amazon-meta", default=defaults / "amazon" / "meta_Automotive.sample.jsonl")
    parser.add_argument("--output", default=settings.catalog_path)
    parser.add_argument("--max-models", type=int, default=30)
    parser.add_argument("--listings-per-model", type=int, default=40)
    parser.add_argument("--max-reviews", type=int, default=2_000)
    parser.add_argument("--amazon-records", type=int, default=50_000)
    parser.add_argument("--skip-nhtsa", action="store_true")
    args = parser.parse_args()

    try:
        catalog = build_catalog(args)
    except Exception as error:
        print(f"Data build failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    print(
        f"Wrote {output}: {len(catalog['vehicles'])} vehicles, "
        f"{len(catalog['listings'])} listings, {len(catalog['reviews'])} reviews, "
        f"{len(catalog['parts'])} products, {len(catalog['recalls'])} recalls."
    )


if __name__ == "__main__":
    main()
