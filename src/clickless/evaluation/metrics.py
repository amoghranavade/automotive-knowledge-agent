from __future__ import annotations

import math
import random
from statistics import mean
from typing import Callable


def precision_at_k(relevance: list[float], k: int = 5, threshold: float = 2.0) -> float:
    values = relevance[:k]
    return sum(value >= threshold for value in values) / k


def ndcg_at_k(relevance: list[float], k: int = 5) -> float:
    values = relevance[:k]
    dcg = sum((2**value - 1) / math.log2(index + 2) for index, value in enumerate(values))
    ideal = sorted(relevance, reverse=True)[:k]
    idcg = sum((2**value - 1) / math.log2(index + 2) for index, value in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def citation_coverage(candidates: list[dict]) -> float:
    claims = [
        claim
        for candidate in candidates
        for group in (candidate.get("pros", []), candidate.get("cautions", []))
        for claim in group
    ]
    return sum(bool(claim.get("evidence_ids")) for claim in claims) / len(claims) if claims else 0.0


def cohen_kappa(first: list[int], second: list[int]) -> float:
    if len(first) != len(second) or not first:
        raise ValueError("Judge label lists must be non-empty and aligned")
    labels = sorted(set(first) | set(second))
    observed = sum(left == right for left, right in zip(first, second)) / len(first)
    expected = sum(
        (first.count(label) / len(first)) * (second.count(label) / len(second))
        for label in labels
    )
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def paired_bootstrap_ci(
    first: list[float],
    second: list[float],
    statistic: Callable[[list[float]], float] = mean,
    samples: int = 2_000,
    seed: int = 42,
) -> tuple[float, float, float]:
    if len(first) != len(second) or not first:
        raise ValueError("Paired samples must be non-empty and aligned")
    generator = random.Random(seed)
    differences = []
    for _ in range(samples):
        indexes = [generator.randrange(len(first)) for _ in first]
        differences.append(
            statistic([first[index] for index in indexes])
            - statistic([second[index] for index in indexes])
        )
    differences.sort()
    return (
        statistic(first) - statistic(second),
        differences[int(samples * 0.025)],
        differences[min(int(samples * 0.975), samples - 1)],
    )
