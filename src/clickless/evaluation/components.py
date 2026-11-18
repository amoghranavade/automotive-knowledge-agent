from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import random
import threading
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from clickless.config import settings
from clickless.data import load_catalog
from clickless.graph import build_graph, validate_rdf, write_rdf
from clickless.llm import QwenClient, QwenError, parse_request_with_model
from clickless.models import Constraints
from clickless.parser import parse


SLOTS = ("max_price", "max_mileage", "min_year", "body_style", "drivetrain")
NUMERIC_SLOTS = ("max_price", "max_mileage", "min_year")

_SLOT_SPECS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("Looking for an AWD crossover 2008+ under $18.5k.", {"max_price": 18_500, "max_mileage": None, "min_year": 2008, "body_style": "suv", "drivetrain": "AWD"}),
    ("Find an all-wheel drive SUV under $18k with below 95k miles.", {"max_price": 18_000, "max_mileage": 95_000, "min_year": None, "body_style": "suv", "drivetrain": "AWD"}),
    ("Need a 4x4 pickup at most $14,500 with under 110,000 miles.", {"max_price": 14_500, "max_mileage": 110_000, "min_year": None, "body_style": "truck", "drivetrain": "4WD"}),
    ("Sedan 2010 or newer, budget of $12,000, under 80k miles.", {"max_price": 12_000, "max_mileage": 80_000, "min_year": 2010, "body_style": "sedan", "drivetrain": None}),
    ("FWD sedan after 2008, $11,500 or less, less than 90,000 miles.", {"max_price": 11_500, "max_mileage": 90_000, "min_year": 2008, "body_style": "sedan", "drivetrain": "FWD"}),
    ("Reliable AWD crossover at least 2009 up to $21,000 with max 75k miles.", {"max_price": 21_000, "max_mileage": 75_000, "min_year": 2009, "body_style": "suv", "drivetrain": "AWD"}),
    ("Looking for a 4WD truck 2007+ no more than $19k and below 100k miles.", {"max_price": 19_000, "max_mileage": 100_000, "min_year": 2007, "body_style": "truck", "drivetrain": "4WD"}),
    ("RWD sedan from 2011 under $24,500 with under 65,000 miles.", {"max_price": 24_500, "max_mileage": 65_000, "min_year": 2011, "body_style": "sedan", "drivetrain": "RWD"}),
    ("SUV below $9,500 with under 120k miles.", {"max_price": 9_500, "max_mileage": 120_000, "min_year": None, "body_style": "suv", "drivetrain": None}),
    ("Pickup truck 2009 or newer under $25k for winter driving.", {"max_price": 25_000, "max_mileage": None, "min_year": 2009, "body_style": "truck", "drivetrain": None}),
    ("Four-wheel drive SUV 2008 or newer under $16,500 with under 95,000 mi.", {"max_price": 16_500, "max_mileage": 95_000, "min_year": 2008, "body_style": "suv", "drivetrain": "4WD"}),
    ("Front-wheel drive sedan under $10k with less than 85k miles.", {"max_price": 10_000, "max_mileage": 85_000, "min_year": None, "body_style": "sedan", "drivetrain": "FWD"}),
    ("AWD sedan 2010+ at most $17,500 and up to 70,000 miles.", {"max_price": 17_500, "max_mileage": 70_000, "min_year": 2010, "body_style": "sedan", "drivetrain": "AWD"}),
    ("Crossover under $13.5k. with under 105k miles.", {"max_price": 13_500, "max_mileage": 105_000, "min_year": None, "body_style": "suv", "drivetrain": None}),
    ("4x4 SUV after 2009, maximum $22,000, below 80,000 miles.", {"max_price": 22_000, "max_mileage": 80_000, "min_year": 2009, "body_style": "suv", "drivetrain": "4WD"}),
    ("Rear-wheel drive pickup 2007 or newer under $15,000.", {"max_price": 15_000, "max_mileage": None, "min_year": 2007, "body_style": "truck", "drivetrain": "RWD"}),
    ("Budget of $8,000 for a reliable sedan under 130,000 miles.", {"max_price": 8_000, "max_mileage": 130_000, "min_year": None, "body_style": "sedan", "drivetrain": None}),
    ("All wheel drive crossover 2012 or newer under $27k with under 60k miles.", {"max_price": 27_000, "max_mileage": 60_000, "min_year": 2012, "body_style": "suv", "drivetrain": "AWD"}),
    ("4WD pickup 2008+ with $23,000 or less and under 100,000 miles.", {"max_price": 23_000, "max_mileage": 100_000, "min_year": 2008, "body_style": "truck", "drivetrain": "4WD"}),
    ("FWD crossover newer than 2009 below $16k and below 90k miles.", {"max_price": 16_000, "max_mileage": 90_000, "min_year": 2009, "body_style": "suv", "drivetrain": "FWD"}),
    ("Sedan at least 2011 under $14,000 with no more than 75k miles.", {"max_price": 14_000, "max_mileage": 75_000, "min_year": 2011, "body_style": "sedan", "drivetrain": None}),
    ("Truck up to $29,500, 2010 or newer, under 85,000 miles.", {"max_price": 29_500, "max_mileage": 85_000, "min_year": 2010, "body_style": "truck", "drivetrain": None}),
    ("AWD SUV under $19.5k. with strong safety and low repair costs.", {"max_price": 19_500, "max_mileage": None, "min_year": None, "body_style": "suv", "drivetrain": "AWD"}),
    ("4WD truck below 115k miles and at most $20,500.", {"max_price": 20_500, "max_mileage": 115_000, "min_year": None, "body_style": "truck", "drivetrain": "4WD"}),
    ("FWD sedan 2007+ under $9k with under 110,000 miles.", {"max_price": 9_000, "max_mileage": 110_000, "min_year": 2007, "body_style": "sedan", "drivetrain": "FWD"}),
    ("Find an SUV 2009 and newer under $18k with under 95k miles.", {"max_price": 18_000, "max_mileage": 95_000, "min_year": 2009, "body_style": "suv", "drivetrain": None}),
    ("Need a 4x4 pickup from 2010 under $26,000 with under 80,000 miles.", {"max_price": 26_000, "max_mileage": 80_000, "min_year": 2010, "body_style": "truck", "drivetrain": "4WD"}),
    ("All-wheel drive sedan 2008+ less than $15.5k. with below 100k miles.", {"max_price": 15_500, "max_mileage": 100_000, "min_year": 2008, "body_style": "sedan", "drivetrain": "AWD"}),
    ("RWD truck under $17,000 with under 90k mi and good brakes.", {"max_price": 17_000, "max_mileage": 90_000, "min_year": None, "body_style": "truck", "drivetrain": "RWD"}),
    ("Crossover 2012+ with a budget of $24,000 and under 65k miles.", {"max_price": 24_000, "max_mileage": 65_000, "min_year": 2012, "body_style": "suv", "drivetrain": None}),
    ("FWD sedan after 2010, maximum $13,000, under 70,000 miles.", {"max_price": 13_000, "max_mileage": 70_000, "min_year": 2010, "body_style": "sedan", "drivetrain": "FWD"}),
    ("AWD crossover 2007 or newer under $12,500 for snowy roads.", {"max_price": 12_500, "max_mileage": None, "min_year": 2007, "body_style": "suv", "drivetrain": "AWD"}),
    ("Pickup 2008+ no more than $21,500 with under 105,000 miles.", {"max_price": 21_500, "max_mileage": 105_000, "min_year": 2008, "body_style": "truck", "drivetrain": None}),
    ("4WD SUV under $31k with under 60,000 miles and cargo space.", {"max_price": 31_000, "max_mileage": 60_000, "min_year": None, "body_style": "suv", "drivetrain": "4WD"}),
    ("Sedan $10,500 or less, 2008 or newer, below 95k miles.", {"max_price": 10_500, "max_mileage": 95_000, "min_year": 2008, "body_style": "sedan", "drivetrain": None}),
    ("RWD sedan 2009+ up to $22,500 and under 75k miles.", {"max_price": 22_500, "max_mileage": 75_000, "min_year": 2009, "body_style": "sedan", "drivetrain": "RWD"}),
    ("All-wheel drive SUV at least 2010 under $25k with below 85,000 miles.", {"max_price": 25_000, "max_mileage": 85_000, "min_year": 2010, "body_style": "suv", "drivetrain": "AWD"}),
    ("4x4 truck 2011 or newer at most $32,000 with under 70k miles.", {"max_price": 32_000, "max_mileage": 70_000, "min_year": 2011, "body_style": "truck", "drivetrain": "4WD"}),
    ("FWD crossover under $14,500 with under 88,000 miles.", {"max_price": 14_500, "max_mileage": 88_000, "min_year": None, "body_style": "suv", "drivetrain": "FWD"}),
    ("Sedan 2007 or newer under $7,500 with under 125k miles.", {"max_price": 7_500, "max_mileage": 125_000, "min_year": 2007, "body_style": "sedan", "drivetrain": None}),
    ("AWD pickup under $28k with less than 80,000 miles.", {"max_price": 28_000, "max_mileage": 80_000, "min_year": None, "body_style": "truck", "drivetrain": "AWD"}),
    ("SUV 2008+ below $18.5k. with good fuel economy.", {"max_price": 18_500, "max_mileage": None, "min_year": 2008, "body_style": "suv", "drivetrain": None}),
    ("FWD sedan below 95k miles and under $11,000.", {"max_price": 11_000, "max_mileage": 95_000, "min_year": None, "body_style": "sedan", "drivetrain": "FWD"}),
    ("4WD crossover 2009 or newer with a budget of $19,000 and under 90k miles.", {"max_price": 19_000, "max_mileage": 90_000, "min_year": 2009, "body_style": "suv", "drivetrain": "4WD"}),
    ("Truck from 2012 under $33k with under 65,000 miles.", {"max_price": 33_000, "max_mileage": 65_000, "min_year": 2012, "body_style": "truck", "drivetrain": None}),
    ("All wheel drive sedan under $20,000 with under 85k miles.", {"max_price": 20_000, "max_mileage": 85_000, "min_year": None, "body_style": "sedan", "drivetrain": "AWD"}),
    ("RWD crossover 2010 or newer at most $23,500 with under 75,000 miles.", {"max_price": 23_500, "max_mileage": 75_000, "min_year": 2010, "body_style": "suv", "drivetrain": "RWD"}),
    ("4x4 pickup under $16,000 with under 120k miles.", {"max_price": 16_000, "max_mileage": 120_000, "min_year": None, "body_style": "truck", "drivetrain": "4WD"}),
    ("SUV newer than 2010 under $26,500 with up to 60k miles.", {"max_price": 26_500, "max_mileage": 60_000, "min_year": 2010, "body_style": "suv", "drivetrain": None}),
    ("FWD sedan 2008 or newer, $12.5k or less, under 95,000 miles.", {"max_price": 12_500, "max_mileage": 95_000, "min_year": 2008, "body_style": "sedan", "drivetrain": "FWD"}),
)


def slot_cases() -> list[dict[str, Any]]:
    return [
        {
            "id": f"slot-{index + 1:02d}",
            "query": query,
            "expected": dict(expected),
        }
        for index, (query, expected) in enumerate(_SLOT_SPECS)
    ]


def _normalize_slot(slot: str, value: Any) -> Any:
    if value is None or value == "":
        return None
    if slot in NUMERIC_SLOTS:
        if isinstance(value, bool):
            return None
        try:
            return int(round(float(str(value).replace(",", ""))))
        except (TypeError, ValueError):
            return None
    text = str(value).strip()
    if slot == "body_style":
        return text.lower()
    if slot == "drivetrain":
        return text.upper()
    return text


def _evaluate_parser(
    cases: list[dict[str, Any]],
    parser_fn: Callable[[str], Constraints],
    max_workers: int = 1,
) -> dict[str, Any]:
    if max_workers > 1 and len(cases) > 1:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(cases))) as executor:
            predictions = list(executor.map(parser_fn, [case["query"] for case in cases]))
    else:
        predictions = [parser_fn(case["query"]) for case in cases]
    records = []
    for case, predicted in zip(cases, predictions):
        actual = predicted.to_dict() if hasattr(predicted, "to_dict") else dict(predicted)
        norm_expected = {
            slot: _normalize_slot(slot, case["expected"].get(slot)) for slot in SLOTS
        }
        norm_predicted = {
            slot: _normalize_slot(slot, actual.get(slot)) for slot in SLOTS
        }
        records.append(
            {
                "id": case["id"],
                "query": case["query"],
                "expected": norm_expected,
                "predicted": norm_predicted,
            }
        )
    per_slot = {}
    for slot in SLOTS:
        truth = [str(item["expected"][slot]) for item in records]
        predicted = [str(item["predicted"][slot]) for item in records]
        per_slot[slot] = _macro_f1(truth, predicted)
    return {
        "cases": len(records),
        "exact_match": mean(
            all(item["expected"][name] == item["predicted"][name] for name in SLOTS)
            for item in records
        ),
        "normalized_numeric_exact_match": mean(
            all(
                item["expected"][name] == item["predicted"][name]
                for name in NUMERIC_SLOTS
            )
            for item in records
        ),
        "macro_f1": mean(per_slot.values()),
        "per_slot_f1": per_slot,
        "records": records,
    }


def slot_metrics(
    use_model: bool = False,
    client: Any = None,
) -> dict[str, Any]:
    cases = slot_cases()
    rule_metrics = _evaluate_parser(cases, lambda query: parse(query)[0])
    result: dict[str, Any] = {
        "cases": rule_metrics["cases"],
        "exact_match": rule_metrics["exact_match"],
        "normalized_numeric_exact_match": rule_metrics["normalized_numeric_exact_match"],
        "macro_f1": rule_metrics["macro_f1"],
        "per_slot_f1": rule_metrics["per_slot_f1"],
        "records": rule_metrics["records"],
        "rule_parser": rule_metrics,
    }
    if use_model:
        active_client = client
        model_error: str | None = None
        if active_client is None:
            try:
                active_client = QwenClient.from_env()
            except QwenError as error:
                model_error = str(error)
        if active_client is None:
            result["model_parser"] = {
                "available": False,
                "error": model_error or "Qwen client is not configured",
            }
        else:
            errors = 0
            error_lock = threading.Lock()

            def _parse_with_qwen(query: str) -> Constraints:
                nonlocal errors
                fallback, questions = parse(query)
                try:
                    constraints, _ = parse_request_with_model(
                        active_client,
                        query,
                        fallback,
                        questions,
                    )
                    return constraints
                except (QwenError, TypeError, ValueError, KeyError):
                    with error_lock:
                        errors += 1
                    return fallback

            model_metrics = _evaluate_parser(cases, _parse_with_qwen, max_workers=6)
            model_metrics["available"] = True
            model_metrics["fallback_errors"] = errors
            result["model_parser"] = model_metrics
    return result


def _strip_records(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "records"}


def run(
    output: Path,
    overwrite: bool = False,
    use_model: bool = False,
    client: Any = None,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    slots = slot_metrics(use_model=use_model, client=client)
    (output / "slot_parsing_metrics.json").write_text(
        json.dumps(slots, indent=2), encoding="utf-8"
    )
    graph_path = output / "graph_link_audit.csv"
    web_path = output / "web_extraction_audit.csv"
    catalog = load_catalog()
    if overwrite or not graph_path.exists():
        _write_graph_audit(graph_path, catalog=catalog)
    if overwrite or not web_path.exists():
        _write_web_audit(web_path)
    ttl_path = settings.processed_path / "clickless.ttl"
    if overwrite or not ttl_path.exists():
        write_rdf(catalog, ttl_path)
    conforms, _ = validate_rdf(
        ttl_path,
        settings.root / "data" / "shapes.ttl",
    )
    slot_summary: dict[str, Any] = {
        key: value
        for key, value in slots.items()
        if key not in {"records", "rule_parser", "model_parser"}
    }
    slot_summary["rule_parser"] = _strip_records(slots["rule_parser"])
    if "model_parser" in slots:
        slot_summary["model_parser"] = _strip_records(slots["model_parser"])

    result = {
        "slot_parsing": slot_summary,
        "shacl_conforms": conforms,
        "graph_link_audit": _labeled_audit_metrics(graph_path),
        "web_extraction_audit": _web_audit_metrics(web_path),
    }
    (output / "component_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def _write_graph_audit(path: Path, catalog: dict[str, Any] | None = None) -> None:
    graph = build_graph(catalog if catalog is not None else load_catalog())
    relations = {
        "LISTING_OF",
        "SHARES_GENERATION",
        "REVIEWS",
        "HAS_ASSERTION",
        "MENTIONS_COMPONENT",
        "AFFECTED_BY",
        "AFFECTS_COMPONENT",
        "COMPATIBLE_WITH",
        "RELATED_PRODUCT",
        "HAS_RATING",
    }
    edges = [item for item in graph["edges"] if item["relation"] in relations]
    sample = random.Random(42).sample(edges, min(100, len(edges)))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["start", "relation", "end", "valid", "notes"],
        )
        writer.writeheader()
        for edge in sample:
            writer.writerow(
                {
                    "start": edge["start"],
                    "relation": edge["relation"],
                    "end": edge["end"],
                }
            )


def _write_web_audit(path: Path) -> None:
    capture_root = settings.root / "data" / "cache" / "retail"
    captures = []
    if capture_root.exists():
        for source in sorted(capture_root.glob("*.json"))[:100]:
            try:
                payload = json.loads(source.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            products = payload.get("products") if isinstance(payload, dict) else []
            if not isinstance(products, list):
                products = []
            captures.append((source.stem, payload, products))
    with path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "capture_id",
            "site",
            "url",
            "captured_at",
            "extracted_records",
            "complete_name_rate",
            "complete_url_rate",
            "complete_price_rate",
            "gold_correct_records",
            "gold_total_records",
            "notes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for capture_id, payload, products in captures:
            total = len(products)
            writer.writerow(
                {
                    "capture_id": capture_id,
                    "site": payload.get("site"),
                    "url": payload.get("url"),
                    "captured_at": payload.get("captured_at"),
                    "extracted_records": total,
                    "complete_name_rate": _completeness(products, "name"),
                    "complete_url_rate": _completeness(products, "url"),
                    "complete_price_rate": _completeness(products, "price"),
                }
            )


def _labeled_audit_metrics(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    labels = [row["valid"].strip().lower() for row in rows if row["valid"].strip()]
    return {
        "sampled_links": len(rows),
        "labeled_links": len(labels),
        "precision": (
            sum(value in {"1", "true", "yes", "valid"} for value in labels) / len(labels)
            if labels
            else None
        ),
    }


def _web_audit_metrics(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    correct = sum(_integer(row.get("gold_correct_records")) or 0 for row in rows)
    total = sum(_integer(row.get("gold_total_records")) or 0 for row in rows)
    return {
        "captured_pages": len(rows),
        "audited_records": total,
        "field_precision": correct / total if total else None,
    }


def _completeness(records: list[dict[str, Any]], field: str) -> float | None:
    return (
        sum(item.get(field) not in {None, ""} for item in records) / len(records)
        if records
        else None
    )


def _macro_f1(truth: list[str], predicted: list[str]) -> float:
    scores = []
    for label in sorted(set(truth) | set(predicted)):
        tp = sum(a == label and b == label for a, b in zip(truth, predicted))
        fp = sum(a != label and b == label for a, b in zip(truth, predicted))
        fn = sum(a == label and b != label for a, b in zip(truth, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return mean(scores)


def _integer(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and analyze ClickLess component audits")
    parser.add_argument("--output", default=settings.root / "evaluation" / "output")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use-model", action="store_true", help="Also score Qwen slot parsing")
    args = parser.parse_args()
    print(json.dumps(run(Path(args.output), args.overwrite, use_model=args.use_model), indent=2))


if __name__ == "__main__":
    main()
