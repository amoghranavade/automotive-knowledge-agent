from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from clickless.config import settings
from clickless.data import load_catalog
from clickless.graph import build_graph, validate_rdf
from clickless.parser import parse


SLOTS = ("max_price", "max_mileage", "min_year", "body_style", "drivetrain")


def slot_cases() -> list[dict[str, Any]]:
    bodies = ("SUV", "sedan", "truck")
    drives = ("AWD", "4WD", "FWD", "RWD")
    prices = (9000, 12000, 18000, 25000, 32000)
    mileages = (60000, 80000, 100000, 120000)
    years = (2007, 2008, 2009, 2010, 2012)
    cases = []
    for index in range(50):
        body = bodies[index % len(bodies)]
        drive = drives[index % len(drives)]
        price = prices[index % len(prices)]
        mileage = mileages[index % len(mileages)]
        year = years[index % len(years)]
        article = "an" if body == "SUV" else "a"
        templates = (
            f"Find {article} {drive} {body} {year} or newer under ${price:,} with under {mileage:,} miles.",
            f"I need {article} {body} with {drive}, budget ${price // 1000}k, less than {mileage // 1000}k miles, {year} or newer.",
            f"Looking for a {year}+ {drive} {body}; max ${price:,}; mileage below {mileage:,} mi.",
            f"Show me {drive} {body}s from {year} on, no more than {mileage:,} miles, up to ${price:,}.",
            f"{body} please: {drive}, ${price:,} max, under {mileage // 1000}k miles, {year} and newer.",
        )
        query = templates[index % len(templates)]
        cases.append(
            {
                "id": f"slot-{index + 1:02d}",
                "query": query,
                "expected": {
                    "max_price": price,
                    "max_mileage": mileage,
                    "min_year": year,
                    "body_style": body.lower(),
                    "drivetrain": drive,
                },
            }
        )
    return cases


def slot_metrics(client: Any = None) -> dict[str, Any]:
    records = []
    for case in slot_cases():
        predicted, questions = parse(case["query"])
        if client is not None:
            from clickless.llm import parse_request_with_model

            predicted, _ = parse_request_with_model(client, case["query"], predicted, questions)
        actual = predicted.to_dict()
        records.append(
            {
                "id": case["id"],
                "query": case["query"],
                "expected": case["expected"],
                "predicted": {name: actual[name] for name in SLOTS},
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
        "macro_f1": mean(per_slot.values()),
        "per_slot_f1": per_slot,
        "records": records,
    }


def run(output: Path, overwrite: bool = False, use_model: bool = False) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    slots = slot_metrics()
    model_slots = None
    if use_model:
        from clickless.llm import QwenClient

        model_slots = slot_metrics(QwenClient.from_env())
    (output / "slot_parsing_metrics.json").write_text(
        json.dumps(slots, indent=2), encoding="utf-8"
    )
    graph_path = output / "graph_link_audit.csv"
    web_path = output / "web_extraction_audit.csv"
    if overwrite or not graph_path.exists():
        _write_graph_audit(graph_path)
    if overwrite or not web_path.exists():
        _write_web_audit(web_path)
    conforms, _ = validate_rdf(
        settings.processed_path / "clickless.ttl",
        settings.root / "data" / "shapes.ttl",
    )
    result = {
        "slot_parsing": {
            key: value for key, value in slots.items() if key != "records"
        },
        "slot_parsing_qwen": (
            {key: value for key, value in model_slots.items() if key != "records"}
            if model_slots
            else None
        ),
        "shacl_conforms": conforms,
        "graph_link_audit": _labeled_audit_metrics(graph_path),
        "web_extraction_audit": _web_audit_metrics(web_path),
    }
    (output / "component_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def _write_graph_audit(path: Path) -> None:
    graph = build_graph(load_catalog())
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
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
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
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
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
    print(json.dumps(run(Path(args.output), args.overwrite, args.use_model), indent=2))


if __name__ == "__main__":
    main()
