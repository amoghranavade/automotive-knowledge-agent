from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from statistics import mean, median
from typing import Any

from clickless.config import settings
from clickless.data import load_catalog
from clickless.evaluation.metrics import citation_coverage
from clickless.evaluation.scenarios import default_scenarios
from clickless.service import recommend


def _p95(latencies: list[float]) -> float:
    if not latencies:
        return 0.0
    ordered = sorted(latencies)
    index = min(max(math.ceil(len(ordered) * 0.95) - 1, 0), len(ordered) - 1)
    return ordered[index]


def _candidate_satisfies_constraints(
    candidate: dict[str, Any], expected: dict[str, Any]
) -> bool:
    max_price = expected.get("max_price")
    if max_price is not None:
        price = candidate.get("price")
        if price is None or price > max_price:
            return False
    max_mileage = expected.get("max_mileage")
    if max_mileage is not None:
        mileage = candidate.get("mileage")
        if mileage is None or mileage > max_mileage:
            return False
    min_year = expected.get("min_year")
    if min_year is not None:
        year = candidate.get("year")
        if year is None or year < min_year:
            return False
    body_style = expected.get("body_style")
    if body_style:
        if str(candidate.get("body_style") or "").casefold() != str(body_style).casefold():
            return False
    drivetrain = expected.get("drivetrain")
    if drivetrain:
        if str(candidate.get("drivetrain") or "").casefold() != str(drivetrain).casefold():
            return False
    make = expected.get("make")
    if make:
        if str(candidate.get("make") or "").casefold() != str(make).casefold():
            return False
    return True


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(record["latency_seconds"]) for record in records]
    gold_checks = [
        bool(record["gold_constraint_satisfied"])
        for record in records
        if record.get("gold_constraint_satisfied") is not None
    ]
    abstention_checks = [
        bool(record["abstained_correctly"])
        for record in records
        if record.get("abstained_correctly") is not None
    ]
    retentions = [
        float(record["cohort_retention"])
        for record in records
        if record.get("cohort_retention") is not None
    ]
    return {
        "runs": len(records),
        "median_latency_seconds": median(latencies) if latencies else 0.0,
        "p95_latency_seconds": _p95(latencies),
        "constraint_satisfaction": (
            sum(bool(record["constraint_satisfied"]) for record in records) / len(records)
            if records
            else 0.0
        ),
        "gold_constraint_satisfaction": (
            sum(gold_checks) / len(gold_checks) if gold_checks else None
        ),
        "abstention_accuracy": (
            sum(abstention_checks) / len(abstention_checks)
            if abstention_checks
            else None
        ),
        "mean_cohort_retention": mean(retentions) if retentions else None,
        "citation_coverage": (
            sum(float(record["citation_coverage"]) for record in records) / len(records)
            if records
            else 0.0
        ),
    }


def run(
    configurations: str,
    use_model: bool,
    output: Path,
    scenarios: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    selected_scenarios = scenarios if scenarios is not None else default_scenarios()
    records = []
    for scenario in selected_scenarios:
        answerable = bool(scenario.get("answerable", True))
        expected = dict(scenario.get("expected_constraints") or {})
        for configuration in configurations:
            start = time.perf_counter()
            result = recommend(
                scenario["query"],
                live_nhtsa=False,
                use_model=use_model,
                configuration=configuration,
                candidate_limit=10,
            )
            latency = time.perf_counter() - start
            candidates = result.get("candidates") or []
            exact_match = result.get("match_mode") == "exact"
            gold_satisfied = (
                bool(
                    exact_match
                    and candidates
                    and all(
                        _candidate_satisfies_constraints(candidate, expected)
                        for candidate in candidates
                    )
                )
                if answerable
                else None
            )
            abstained = (
                (result.get("match_mode") == "closest_alternatives")
                if not answerable
                else None
            )
            records.append(
                {
                    "scenario_id": scenario["id"],
                    "split": scenario["split"],
                    "query": scenario["query"],
                    "answerable": answerable,
                    "expected_constraints": expected,
                    "configuration": configuration,
                    "latency_seconds": round(latency, 6),
                    "citation_coverage": citation_coverage(candidates),
                    "constraint_satisfied": exact_match,
                    "gold_constraint_satisfied": gold_satisfied,
                    "abstained_correctly": abstained,
                    "cohort_retention": result.get("cohort_retention"),
                    "candidate_ids": [item["id"] for item in candidates],
                    "result": result,
                }
            )

    results_path = output / "system_results.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    _write_judging_template(records, output / "human_judgments.csv")
    blinding_key = json.loads((output / "blinding_key.json").read_text(encoding="utf-8"))
    system_labels = {configuration: label for label, configuration in blinding_key.items()}
    _write_citation_audit(records, system_labels, output / "citation_audit.csv")
    _write_explanation_ratings(records, system_labels, output / "explanation_ratings.csv")
    _write_absa_template(output / "absa_annotations.csv")

    aggregate = _summarize_records(records)
    summary = {
        **aggregate,
        "configurations": list(configurations),
        "by_configuration": {
            configuration: _summarize_records(
                [record for record in records if record["configuration"] == configuration]
            )
            for configuration in configurations
        },
        "human_metrics_pending": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _write_judging_template(records: list[dict[str, Any]], path: Path) -> None:
    configurations = sorted({record["configuration"] for record in records})
    labels = [f"system-{index + 1}" for index in range(len(configurations))]
    random.Random(42).shuffle(labels)
    blinded = dict(zip(configurations, labels))
    path.with_name("blinding_key.json").write_text(
        json.dumps({label: configuration for configuration, label in blinded.items()}, indent=2),
        encoding="utf-8",
    )
    seen = set()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["scenario_id", "system_id", "candidate_id", "rank", "judge_1_score", "judge_2_score", "notes"],
        )
        writer.writeheader()
        for record in records:
            for rank, candidate_id in enumerate(record["candidate_ids"], 1):
                key = (record["scenario_id"], record["configuration"], candidate_id)
                if key in seen:
                    continue
                seen.add(key)
                writer.writerow(
                    {
                        "scenario_id": record["scenario_id"],
                        "system_id": blinded[record["configuration"]],
                        "candidate_id": candidate_id,
                        "rank": rank,
                    }
                )


def _write_absa_template(path: Path) -> None:
    catalog = load_catalog()
    owner = [item for item in catalog["reviews"] if item.get("source_kind") == "owner_review"]
    product = [item for item in catalog["reviews"] if item.get("source_kind") == "product_review"]
    generator = random.Random(42)
    product_sample = generator.sample(product, min(50, len(product)))
    owner_sample = generator.sample(owner, min(300 - len(product_sample), len(owner)))
    selected = owner_sample + product_sample
    generator.shuffle(selected)
    rows = [
        (
            review["source_id"],
            str(review.get("text") or "").strip()[:1_500],
            review.get("source_kind", "owner_review"),
        )
        for review in selected
        if str(review.get("text") or "").strip()
    ][:300]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_id",
                "text",
                "source_kind",
                "split",
                "aspect",
                "subsystem",
                "polarity",
                "span",
                "annotator",
                "double_annotated",
            ],
        )
        writer.writeheader()
        for index, (source_id, text, source_kind) in enumerate(rows):
            double = index < 120
            split = "train" if index < 180 else "development" if index < 240 else "test"
            writer.writerow(
                {
                    "source_id": source_id,
                    "text": text,
                    "source_kind": source_kind,
                    "split": split,
                    "annotator": "1",
                    "double_annotated": double,
                }
            )
            if double:
                writer.writerow(
                    {
                        "source_id": source_id,
                        "text": text,
                        "source_kind": source_kind,
                        "split": split,
                        "annotator": "2",
                        "double_annotated": True,
                    }
                )


def _write_citation_audit(
    records: list[dict[str, Any]],
    system_labels: dict[str, str],
    path: Path,
) -> None:
    claims = []
    for record in records:
        for candidate in record["result"]["candidates"]:
            for group in ("pros", "cautions"):
                for claim in candidate.get(group, []):
                    claims.append(
                        {
                            "scenario_id": record["scenario_id"],
                            "system_id": system_labels[record["configuration"]],
                            "candidate_id": candidate["id"],
                            "claim": claim["text"],
                            "evidence_ids": "|".join(claim["evidence_ids"]),
                        }
                    )
    sample = random.Random(42).sample(claims, min(100, len(claims)))
    with path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "scenario_id",
            "system_id",
            "candidate_id",
            "claim",
            "evidence_ids",
            "correct_support",
            "notes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sample)


def _write_explanation_ratings(
    records: list[dict[str, Any]],
    system_labels: dict[str, str],
    path: Path,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "scenario_id",
            "system_id",
            "candidate_ids",
            "judge_1_clarity",
            "judge_2_clarity",
            "judge_1_tradeoff_completeness",
            "judge_2_tradeoff_completeness",
            "notes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "scenario_id": record["scenario_id"],
                    "system_id": system_labels[record["configuration"]],
                    "candidate_ids": "|".join(record["candidate_ids"][:5]),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ClickLess A-E evaluation")
    parser.add_argument("--configurations", default="ABCDE")
    parser.add_argument("--use-model", action="store_true")
    parser.add_argument("--output", default=settings.root / "evaluation" / "output")
    args = parser.parse_args()
    configurations = "".join(dict.fromkeys(args.configurations.upper()))
    if not configurations or any(item not in "ABCDE" for item in configurations):
        raise SystemExit("Configurations must contain only A, B, C, D, and E")
    print(json.dumps(run(configurations, args.use_model, Path(args.output)), indent=2))


if __name__ == "__main__":
    main()
