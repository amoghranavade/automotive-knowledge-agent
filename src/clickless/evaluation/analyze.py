from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from clickless.config import settings
from clickless.evaluation.metrics import cohen_kappa, ndcg_at_k, paired_bootstrap_ci, precision_at_k
from clickless.sentiment import TfidfLogisticSentiment, vader_labels


def _score(value: str) -> int | None:
    try:
        score = int(value)
    except (TypeError, ValueError):
        return None
    if score not in {0, 1, 2, 3}:
        raise ValueError("Human relevance scores must be integers from 0 to 3")
    return score


def analyze(path: Path, use_model: bool = False) -> dict[str, Any]:
    judgments = path / "human_judgments.csv"
    rows = list(csv.DictReader(judgments.open(encoding="utf-8")))
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
    metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (_, configuration), values in groups.items():
        relevance = [score for _, score in sorted(values)]
        metrics[configuration]["precision_at_5"].append(precision_at_k(relevance, 5))
        metrics[configuration]["ndcg_at_5"].append(ndcg_at_k(relevance, 5))
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
                name: mean(values) if values else None
                for name, values in configuration_metrics.items()
            }
            for configuration, configuration_metrics in metrics.items()
        },
    }
    if "C" in metrics:
        comparisons = {}
        for baseline in "ABDE":
            if baseline not in metrics:
                continue
            first = metrics["C"]["ndcg_at_5"]
            second = metrics[baseline]["ndcg_at_5"]
            if len(first) == len(second) and first:
                difference, lower, upper = paired_bootstrap_ci(first, second)
                comparisons[f"C_vs_{baseline}"] = {
                    "ndcg_difference": difference,
                    "ci_95": [lower, upper],
                }
        summary["paired_bootstrap"] = comparisons
    summary["absa"] = _analyze_absa(path / "absa_annotations.csv", use_model)
    summary["citation_precision"] = _citation_precision(path / "citation_audit.csv")
    summary["explanation_ratings"] = _explanation_ratings(
        path / "explanation_ratings.csv"
    )
    return summary


def _tokens(text: str) -> list[str]:
    import re

    return re.findall(r"[a-z0-9]+", str(text or "").casefold())


def _span_f1(rows: list[dict[str, str]], predictions: list[dict[str, str]]) -> float | None:
    """Token-overlap F1 between gold and predicted opinion spans (SemEval-style, relaxed)."""
    scores = []
    for row, predicted in zip(rows, predictions):
        gold = _tokens(row.get("span", ""))
        guess = _tokens(predicted.get("span", ""))
        if not gold:
            continue
        overlap = sum(min(gold.count(token), guess.count(token)) for token in set(gold))
        if not overlap:
            scores.append(0.0)
            continue
        precision, recall = overlap / len(guess), overlap / len(gold)
        scores.append(2 * precision * recall / (precision + recall))
    return mean(scores) if scores else None


def _lexicon_prediction(text: str) -> dict[str, str]:
    from clickless.data.aspects import extract_aspects

    found = extract_aspects(text, "eval")
    if not found:
        return {"aspect": "none", "polarity": "neutral", "span": ""}
    first = found[0]
    return {"aspect": first["aspect"], "polarity": first["polarity"], "span": first["span"]}


def _qwen_predictions(client: Any, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    system = (
        "For each sentence return JSON with the main aspect (reliability, fuel_economy, cargo, "
        "maintenance, or none), its polarity (positive, negative, neutral, mixed) and the exact "
        "opinion span copied from the sentence."
    )
    predictions: dict[str, dict[str, str]] = {}
    for start in range(0, len(rows), 20):
        batch = rows[start : start + 20]
        payload = client.complete_json(
            system,
            json.dumps(
                {
                    "sentences": [{"id": row["sentence_id"], "text": row["text"]} for row in batch],
                    "output": {"sentences": [{"id": "string", "aspect": "string", "polarity": "string", "span": "string"}]},
                }
            ),
            max_tokens=1800,
        )
        for item in payload.get("sentences", []) if isinstance(payload, dict) else []:
            if isinstance(item, dict):
                predictions[str(item.get("id"))] = {
                    "aspect": str(item.get("aspect", "none")),
                    "polarity": str(item.get("polarity", "neutral")),
                    "span": str(item.get("span", "")),
                }
    return [
        predictions.get(row["sentence_id"], {"aspect": "none", "polarity": "neutral", "span": ""})
        for row in rows
    ]


def _analyze_absa(path: Path, use_model: bool = False) -> dict[str, Any]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_source[row.get("sentence_id") or row["source_id"]].append(row)
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
    lexicon = [_lexicon_prediction(row["text"]) for row in test]
    result["baselines"]["lexicon_aspect"] = _classification_metrics(
        [row["aspect"] for row in test], [item["aspect"] for item in lexicon]
    )
    result["baselines"]["lexicon_polarity"] = _classification_metrics(
        [row["polarity"] for row in test], [item["polarity"] for item in lexicon]
    )
    result["baselines"]["lexicon_span_f1"] = _span_f1(test, lexicon)
    if use_model:
        from clickless.llm import QwenClient

        qwen = _qwen_predictions(QwenClient.from_env(), test)
        result["neural"] = {
            "qwen_aspect": _classification_metrics([row["aspect"] for row in test], [item["aspect"] for item in qwen]),
            "qwen_polarity": _classification_metrics([row["polarity"] for row in test], [item["polarity"] for item in qwen]),
            "qwen_span_f1": _span_f1(test, qwen),
        }
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


def _classification_metrics(truth: list[str], predicted: list[str]) -> dict[str, float]:
    labels = sorted(set(truth) | set(predicted))
    accuracy = sum(left == right for left, right in zip(truth, predicted)) / len(truth)
    f1_scores = []
    for label in labels:
        true_positive = sum(left == label and right == label for left, right in zip(truth, predicted))
        false_positive = sum(left != label and right == label for left, right in zip(truth, predicted))
        false_negative = sum(left == label and right != label for left, right in zip(truth, predicted))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1_scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {"accuracy": accuracy, "macro_f1": mean(f1_scores)}


def _citation_precision(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"audited_claims": 0, "precision": None}
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
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
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
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
    parser.add_argument("--use-model", action="store_true", help="Also score Qwen ABSA on the test split")
    args = parser.parse_args()
    root = Path(args.input)
    result = analyze(root, args.use_model)
    output = root / "human_metrics.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
