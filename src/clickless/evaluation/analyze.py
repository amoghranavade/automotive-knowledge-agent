from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from clickless.config import settings
from clickless.data.aspects import extract_aspects
from clickless.evaluation.metrics import cohen_kappa, ndcg_at_k, paired_bootstrap_ci, precision_at_k
from clickless.llm import QwenClient, QwenError
from clickless.sentiment import TfidfLogisticSentiment, vader_labels


POLARITY_CLASSES = ("positive", "negative", "neutral", "mixed")
ASPECT_CLASSES = ("reliability", "fuel_economy", "cargo", "maintenance")

_ABSA_FEW_SHOT_EXAMPLES = [
    {
        "text": "The engine has been bulletproof for 140k miles, though the brake pads wear quickly.",
        "assertions": [
            {
                "aspect": "reliability",
                "subsystem": "engine",
                "polarity": "positive",
                "span": "engine has been bulletproof for 140k miles",
            },
            {
                "aspect": "maintenance",
                "subsystem": "brakes",
                "polarity": "negative",
                "span": "brake pads wear quickly",
            },
        ],
    },
    {
        "text": "Great highway mpg, but city fuel economy drops in winter.",
        "assertions": [
            {
                "aspect": "fuel_economy",
                "subsystem": "fuel_system",
                "polarity": "mixed",
                "span": "Great highway mpg, but city fuel economy drops in winter.",
            }
        ],
    },
    {
        "text": "Scheduled oil service was completed at 60,000 miles.",
        "assertions": [
            {
                "aspect": "maintenance",
                "subsystem": "engine",
                "polarity": "neutral",
                "span": "Scheduled oil service was completed at 60,000 miles.",
            }
        ],
    },
]


def _score(value: str) -> int | None:
    try:
        score = int(value)
    except (TypeError, ValueError):
        return None
    if score not in {0, 1, 2, 3}:
        raise ValueError("Human relevance scores must be integers from 0 to 3")
    return score


def analyze(path: Path, qwen_client: Any = None) -> dict[str, Any]:
    judgments = path / "human_judgments.csv"
    with judgments.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    key_path = path / "blinding_key.json"
    blinding_key = json.loads(key_path.read_text(encoding="utf-8")) if key_path.exists() else {}
    paired = [
        (_score(row["judge_1_score"]), _score(row["judge_2_score"]))
        for row in rows
        if _score(row["judge_1_score"]) is not None and _score(row["judge_2_score"]) is not None
    ]
    groups: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        first = _score(row["judge_1_score"])
        second = _score(row["judge_2_score"])
        available = [value for value in (first, second) if value is not None]
        if not available:
            continue
        configuration = row.get("configuration") or blinding_key.get(row.get("system_id", ""))
        if not configuration:
            continue
        groups[(row["scenario_id"], configuration)].append(
            (int(row["rank"]), mean(available))
        )

    per_scenario: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: {"precision_at_5": {}, "ndcg_at_5": {}}
    )
    for (scenario_id, configuration), values in groups.items():
        relevance = [score for _, score in sorted(values)]
        per_scenario[configuration]["precision_at_5"][scenario_id] = precision_at_k(relevance, 5)
        per_scenario[configuration]["ndcg_at_5"][scenario_id] = ndcg_at_k(relevance, 5)

    summary: dict[str, Any] = {
        "judged_rows": sum(len(values) for values in groups.values()),
        "double_judged_rows": len(paired),
        "cohen_kappa": (
            cohen_kappa([left for left, _ in paired], [right for _, right in paired])
            if paired
            else None
        ),
        "configurations": {
            configuration: {
                name: (
                    mean(metric_map[scenario_id] for scenario_id in sorted(metric_map))
                    if metric_map
                    else None
                )
                for name, metric_map in configuration_metrics.items()
            }
            for configuration, configuration_metrics in sorted(per_scenario.items())
        },
    }
    if "C" in per_scenario:
        comparisons = {}
        c_ndcg = per_scenario["C"]["ndcg_at_5"]
        c_p5 = per_scenario["C"]["precision_at_5"]
        for baseline in "ABDE":
            if baseline not in per_scenario:
                continue
            b_ndcg = per_scenario[baseline]["ndcg_at_5"]
            b_p5 = per_scenario[baseline]["precision_at_5"]
            common_ids = sorted(set(c_ndcg) & set(b_ndcg))
            if not common_ids:
                continue
            ndcg_diff, ndcg_lo, ndcg_hi = paired_bootstrap_ci(
                [c_ndcg[scenario_id] for scenario_id in common_ids],
                [b_ndcg[scenario_id] for scenario_id in common_ids],
            )
            p5_diff, p5_lo, p5_hi = paired_bootstrap_ci(
                [c_p5[scenario_id] for scenario_id in common_ids],
                [b_p5[scenario_id] for scenario_id in common_ids],
            )
            comparisons[f"C_vs_{baseline}"] = {
                "common_scenarios": len(common_ids),
                "ndcg_difference": ndcg_diff,
                "ci_95": [ndcg_lo, ndcg_hi],
                "ndcg_at_5_difference": ndcg_diff,
                "ndcg_at_5_ci_95": [ndcg_lo, ndcg_hi],
                "precision_at_5_difference": p5_diff,
                "precision_at_5_ci_95": [p5_lo, p5_hi],
            }
        summary["paired_bootstrap"] = comparisons
    summary["absa"] = _analyze_absa(path / "absa_annotations.csv", qwen_client=qwen_client)
    summary["citation_precision"] = _citation_precision(path / "citation_audit.csv")
    summary["explanation_ratings"] = _explanation_ratings(
        path / "explanation_ratings.csv"
    )
    return summary


def _dedupe_primary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    primary_seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    for row in rows:
        source_id = str(row.get("source_id") or "")
        annotator = str(row.get("annotator") or "").strip()
        if annotator == "2" and source_id in primary_seen:
            continue
        if source_id and annotator in {"", "1"}:
            primary_seen.add(source_id)
        kept.append(row)
    return kept


def _normalize_span(value: str) -> str:
    return " ".join(str(value or "").casefold().split())


def _span_matches(pred_span: str, gold_span: str) -> bool:
    pred_norm = _normalize_span(pred_span)
    gold_norm = _normalize_span(gold_span)
    if not pred_norm or not gold_norm:
        return False
    if gold_norm in pred_norm or pred_norm in gold_norm:
        return True
    pred_tokens = set(re.findall(r"[a-z0-9]+", pred_norm))
    gold_tokens = set(re.findall(r"[a-z0-9]+", gold_norm))
    if not pred_tokens or not gold_tokens:
        return False
    union = pred_tokens | gold_tokens
    return (len(pred_tokens & gold_tokens) / len(union)) >= 0.5


def _span_metrics(
    gold_by_source: dict[str, list[str]],
    pred_by_source: dict[str, list[str]],
) -> dict[str, Any]:
    tp = 0
    fp = 0
    fn = 0
    evaluated_sources = 0
    for source_id, gold_spans in gold_by_source.items():
        if not gold_spans:
            continue
        evaluated_sources += 1
        preds = pred_by_source.get(source_id, [])
        used: set[int] = set()
        for gold_span in gold_spans:
            matched_index = next(
                (
                    index
                    for index, pred_span in enumerate(preds)
                    if index not in used and _span_matches(pred_span, gold_span)
                ),
                None,
            )
            if matched_index is not None:
                used.add(matched_index)
                tp += 1
            else:
                fn += 1
        fp += len(preds) - len(used)
    if tp + fn == 0:
        return {
            "precision": None,
            "recall": None,
            "f1": None,
            "evaluated_spans": 0,
            "evaluated_sources": 0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
        }
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "evaluated_spans": tp + fn,
        "evaluated_sources": evaluated_sources,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }


def _polarity_metrics(truth: list[str], predicted: list[str]) -> dict[str, Any]:
    base = _classification_metrics(truth, predicted)
    four_class = _classification_metrics(truth, predicted, labels=list(POLARITY_CLASSES))
    return {
        "accuracy": base["accuracy"],
        "macro_f1": base["macro_f1"],
        "polarity_macro_f1_4class": four_class["macro_f1"],
        "per_class_f1": four_class["per_class_f1"],
    }


def _select_assertion_polarity(
    assertions: list[dict[str, Any]],
    gold_aspect: str | None,
    gold_span: str | None,
) -> str:
    if not assertions:
        return "neutral"
    candidates = [
        item for item in assertions if gold_aspect and item.get("aspect") == gold_aspect
    ] or assertions
    if gold_span:
        matched = next(
            (
                item
                for item in candidates
                if _span_matches(str(item.get("span") or ""), gold_span)
            ),
            None,
        )
        if matched is not None:
            return str(matched.get("polarity") or "neutral")
    return str(candidates[0].get("polarity") or "neutral")


def _extract_qwen_absa(
    qwen_client: Any, eval_rows: list[dict[str, Any]]
) -> dict[str, list[dict[str, str]]]:
    unique_texts: dict[str, str] = {}
    for index, row in enumerate(eval_rows):
        source_id = str(row.get("source_id") or f"row-{index}")
        if source_id not in unique_texts:
            unique_texts[source_id] = str(row.get("text") or "")

    reviews = [{"source_id": sid, "text": text} for sid, text in unique_texts.items()]
    system = (
        "Extract aspect-based sentiment assertions as JSON using exact text spans from each review. "
        "Allowed aspects: reliability, fuel_economy, cargo, maintenance. "
        "Allowed polarities: positive, negative, neutral, mixed. "
        "Use 'mixed' when a span expresses both positive and negative sentiment."
    )
    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for start in range(0, len(reviews), 20):
        user = json.dumps(
            {
                "examples": _ABSA_FEW_SHOT_EXAMPLES,
                "reviews": reviews[start : start + 20],
                "output": {
                    "reviews": [
                        {
                            "source_id": "string",
                            "assertions": [
                                {
                                    "aspect": "string",
                                    "subsystem": "string",
                                    "polarity": "string",
                                    "span": "exact review substring",
                                }
                            ],
                        }
                    ]
                },
            }
        )
        payload = qwen_client.complete_json(system, user, max_tokens=1800)
        for item in payload.get("reviews", []):
            if not isinstance(item, dict):
                continue
            source_id = str(item.get("source_id") or "")
            text = unique_texts.get(source_id)
            if text is None:
                continue
            for assertion in item.get("assertions", []):
                if not isinstance(assertion, dict):
                    continue
                aspect = str(assertion.get("aspect") or "").strip()
                polarity = str(assertion.get("polarity") or "").strip()
                span = str(assertion.get("span") or "").strip()
                if (
                    aspect not in ASPECT_CLASSES
                    or polarity not in POLARITY_CLASSES
                    or not span
                    or span.casefold() not in text.casefold()
                ):
                    continue
                by_source[source_id].append(
                    {
                        "aspect": aspect,
                        "subsystem": str(assertion.get("subsystem") or "unknown").strip(),
                        "polarity": polarity,
                        "span": span,
                    }
                )
    return by_source


def evaluate_absa_extractors(
    rows: list[dict[str, Any]],
    qwen_client: Any = None,
) -> dict[str, Any]:
    deduped = [
        row
        for row in _dedupe_primary_rows(rows)
        if str(row.get("text") or "").strip() and str(row.get("polarity") or "").strip()
    ]
    if not deduped:
        return {"evaluated_rows": 0, "extractors": {}}

    has_test = any(row.get("split") == "test" for row in deduped)
    eval_rows = [row for row in deduped if row.get("split") == "test"] if has_test else deduped
    train_rows = [row for row in deduped if row.get("split") == "train"]

    gold_polarities = [str(row["polarity"]).strip().lower() for row in eval_rows]
    gold_by_source: dict[str, list[str]] = defaultdict(list)
    for index, row in enumerate(eval_rows):
        source_id = str(row.get("source_id") or f"row-{index}")
        span = str(row.get("span") or "").strip()
        if span:
            gold_by_source[source_id].append(span)

    # 1. Lexicon extractor (extract_aspects)
    lexicon_by_source: dict[str, list[str]] = defaultdict(list)
    lexicon_polarities: list[str] = []
    lexicon_cache: dict[str, list[dict[str, Any]]] = {}
    for index, row in enumerate(eval_rows):
        source_id = str(row.get("source_id") or f"row-{index}")
        if source_id not in lexicon_cache:
            assertions = extract_aspects(str(row["text"]), source_id)
            lexicon_cache[source_id] = assertions
            seen_spans: set[str] = set()
            for item in assertions:
                span = str(item.get("span") or "").strip()
                norm = _normalize_span(span)
                if span and norm not in seen_spans:
                    seen_spans.add(norm)
                    lexicon_by_source[source_id].append(span)
        lexicon_polarities.append(
            _select_assertion_polarity(
                lexicon_cache[source_id],
                str(row.get("aspect") or "").strip() or None,
                str(row.get("span") or "").strip() or None,
            )
        )

    extractors: dict[str, Any] = {
        "lexicon": {
            "span": _span_metrics(gold_by_source, lexicon_by_source),
            "polarity": _polarity_metrics(gold_polarities, lexicon_polarities),
        }
    }

    # 2. VADER classifier
    eval_texts = [str(row["text"]) for row in eval_rows]
    vader_preds = vader_labels(eval_texts)
    extractors["vader"] = {
        "span": {
            "precision": None,
            "recall": None,
            "f1": None,
            "evaluated_spans": 0,
        },
        "polarity": _polarity_metrics(gold_polarities, vader_preds),
    }

    # 3. TF-IDF Logistic Regression classifier
    train_texts = [str(row["text"]) for row in train_rows]
    train_polarities = [str(row["polarity"]).strip().lower() for row in train_rows]
    tfidf_result: dict[str, Any] = {
        "span": {
            "precision": None,
            "recall": None,
            "f1": None,
            "evaluated_spans": 0,
        },
        "polarity": None,
    }
    if train_texts and len(set(train_polarities)) >= 2:
        try:
            model = TfidfLogisticSentiment().fit(train_texts, train_polarities)
            tfidf_preds = model.predict(eval_texts)
            tfidf_result["polarity"] = _polarity_metrics(gold_polarities, tfidf_preds)
        except ValueError as error:
            tfidf_result["error"] = str(error)
    else:
        tfidf_result["note"] = "Requires train split with at least two polarity classes."
    extractors["tfidf_logistic"] = tfidf_result

    # 4. Optional Qwen few-shot extractor
    if qwen_client is not None:
        try:
            qwen_assertions = _extract_qwen_absa(qwen_client, eval_rows)
            qwen_by_source: dict[str, list[str]] = defaultdict(list)
            qwen_polarities: list[str] = []
            for index, row in enumerate(eval_rows):
                source_id = str(row.get("source_id") or f"row-{index}")
                items = qwen_assertions.get(source_id, [])
                if source_id not in qwen_by_source:
                    seen_q: set[str] = set()
                    for item in items:
                        norm = _normalize_span(item["span"])
                        if norm not in seen_q:
                            seen_q.add(norm)
                            qwen_by_source[source_id].append(item["span"])
                qwen_polarities.append(
                    _select_assertion_polarity(
                        items,
                        str(row.get("aspect") or "").strip() or None,
                        str(row.get("span") or "").strip() or None,
                    )
                )
            extractors["qwen"] = {
                "span": _span_metrics(gold_by_source, qwen_by_source),
                "polarity": _polarity_metrics(gold_polarities, qwen_polarities),
            }
        except (QwenError, ValueError, TypeError, KeyError) as error:
            extractors["qwen"] = {"error": str(error)}

    return {
        "evaluated_rows": len(eval_rows),
        "train_rows": len(train_rows),
        "extractors": extractors,
    }


def _analyze_absa(path: Path, qwen_client: Any = None) -> dict[str, Any]:
    if not path.exists():
        return {"annotated_sentences": 0, "double_annotated_sentences": 0, "baselines": {}}
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_source[row["source_id"]].append(row)
    primary = [
        values[0]
        for values in by_source.values()
        if values[0].get("aspect") and values[0].get("polarity")
    ]
    paired = [
        values[:2]
        for values in by_source.values()
        if len(values) >= 2
        and all(item.get("aspect") and item.get("polarity") for item in values[:2])
    ]
    result: dict[str, Any] = {
        "annotated_sentences": len(primary),
        "double_annotated_sentences": len(paired),
        "aspect_kappa": (
            cohen_kappa(
                [items[0]["aspect"] for items in paired],
                [items[1]["aspect"] for items in paired],
            )
            if paired
            else None
        ),
        "polarity_kappa": (
            cohen_kappa(
                [items[0]["polarity"] for items in paired],
                [items[1]["polarity"] for items in paired],
            )
            if paired
            else None
        ),
        "baselines": {},
    }
    if not primary:
        result["pending"] = "Complete all 300 primary annotation rows before training baselines."
        return result

    result["extractor_evaluation"] = evaluate_absa_extractors(
        primary, qwen_client=qwen_client
    )

    if len(primary) < 300:
        result["pending"] = "Complete all 300 primary annotation rows before training baselines."
        return result
    train = [row for row in primary if row.get("split") == "train"]
    test = [row for row in primary if row.get("split") == "test"]
    if len(train) != 180 or len(test) != 60:
        result["pending"] = "Expected 180 train and 60 test primary annotations."
        return result
    texts = [row["text"] for row in test]
    result["baselines"]["vader_polarity"] = _classification_metrics(
        [row["polarity"] for row in test],
        vader_labels(texts),
    )
    for target in ("polarity", "aspect"):
        model = TfidfLogisticSentiment().fit(
            [row["text"] for row in train],
            [row[target] for row in train],
        )
        result["baselines"][f"tfidf_logistic_{target}"] = _classification_metrics(
            [row[target] for row in test],
            model.predict(texts),
        )
    return result


def _classification_metrics(
    truth: list[str],
    predicted: list[str],
    labels: list[str] | None = None,
) -> dict[str, Any]:
    eval_labels = sorted(labels if labels is not None else (set(truth) | set(predicted)))
    accuracy = sum(left == right for left, right in zip(truth, predicted)) / len(truth)
    per_class: dict[str, float] = {}
    for label in eval_labels:
        true_positive = sum(
            left == label and right == label for left, right in zip(truth, predicted)
        )
        false_positive = sum(
            left != label and right == label for left, right in zip(truth, predicted)
        )
        false_negative = sum(
            left == label and right != label for left, right in zip(truth, predicted)
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        per_class[label] = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    return {
        "accuracy": accuracy,
        "macro_f1": mean(per_class.values()) if per_class else 0.0,
        "per_class_f1": per_class,
    }


def _citation_precision(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"audited_claims": 0, "precision": None}
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    labels = [
        row["correct_support"].strip().lower()
        for row in rows
        if row.get("correct_support", "").strip()
    ]
    return {
        "audited_claims": len(labels),
        "precision": (
            sum(value in {"1", "true", "yes", "correct"} for value in labels)
            / len(labels)
            if labels
            else None
        ),
    }


def _explanation_ratings(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"rated_system_outputs": 0, "clarity": None, "tradeoff_completeness": None}
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    clarity = []
    tradeoffs = []
    for row in rows:
        for field, target in (
            ("judge_1_clarity", clarity),
            ("judge_2_clarity", clarity),
            ("judge_1_tradeoff_completeness", tradeoffs),
            ("judge_2_tradeoff_completeness", tradeoffs),
        ):
            value = row.get(field, "").strip()
            if not value:
                continue
            score = int(value)
            if score not in {1, 2, 3, 4, 5}:
                raise ValueError(f"{field} scores must be integers from 1 to 5")
            target.append(score)
    return {
        "rated_system_outputs": max(len(clarity), len(tradeoffs)) // 2,
        "clarity": mean(clarity) if clarity else None,
        "tradeoff_completeness": mean(tradeoffs) if tradeoffs else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze completed ClickLess human judgments")
    parser.add_argument("--input", default=settings.root / "evaluation" / "output")
    parser.add_argument("--use-model", action="store_true")
    args = parser.parse_args()
    qwen_client = None
    if args.use_model:
        try:
            qwen_client = QwenClient.from_env()
        except QwenError:
            qwen_client = None
    root = Path(args.input)
    result = analyze(root, qwen_client=qwen_client)
    output = root / "human_metrics.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
