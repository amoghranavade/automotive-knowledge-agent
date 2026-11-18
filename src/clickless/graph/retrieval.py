from __future__ import annotations

from collections import defaultdict
import os
from typing import Any

import clickless.config  # noqa: F401

from .build import build_graph, component_node_id, normalize_subsystem
from .queries import run_query_templates


DIRECT_PATHS = {
    "HAS_LISTING": ("listing", "Listing and source"),
    "HAS_RATING": ("safety", "Safety rating and source"),
    "AFFECTED_BY": ("recall", "Recall and affected component"),
    "HAS_REVIEW": ("review", "Owner review and aspect assertion"),
    "COMPATIBLE_WITH": ("product", "Compatible product and component"),
    "RELATED_PRODUCT": ("product", "Related product and component"),
}

EXACT_YEAR_LINAGE_THRESHOLD = 5
SPAN_MAX_CHARS = 140


def retrieve_evidence_paths(
    catalog: dict[str, Any],
    vehicle_id: str,
    max_paths: int = 12,
    aspects: list[str] | None = None,
    live_recalls: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    working_catalog = (
        _catalog_with_live_recalls(catalog, vehicle_id, live_recalls)
        if live_recalls is not None
        else catalog
    )
    backend = os.getenv("CLICKLESS_GRAPH_BACKEND", "memory").strip().lower()
    used_backend = "memory"
    graph = None
    if backend == "neo4j":
        try:
            graph = _neo4j_neighborhood(
                vehicle_id,
                aspects=aspects,
                max_paths=max_paths,
            )
            if live_recalls is not None:
                _overlay_live_recalls_graph(
                    graph,
                    vehicle_id,
                    [
                        item
                        for item in working_catalog.get("recalls", [])
                        if item.get("vehicle_id") == vehicle_id
                    ],
                )
            used_backend = "neo4j"
        except Exception:
            graph = (
                _cached_graph(catalog)
                if live_recalls is None
                else build_graph(working_catalog)
            )
            used_backend = "memory_fallback"
    else:
        graph = (
            _cached_graph(catalog)
            if live_recalls is None
            else build_graph(working_catalog)
        )
    return _retrieve_from_graph(
        graph,
        vehicle_id,
        max_paths,
        used_backend,
        aspects=aspects,
    )


def _catalog_with_live_recalls(
    catalog: dict[str, Any],
    vehicle_id: str,
    live_recalls: list[dict[str, Any]],
) -> dict[str, Any]:
    existing_ids = {
        str(item.get("id"))
        for item in catalog.get("recalls", [])
        if item.get("vehicle_id") != vehicle_id and item.get("id")
    }
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(live_recalls):
        source_id = str(item.get("source_id") or item.get("id") or f"nhtsa-live-{index}")
        base_id = str(item.get("id") or f"{vehicle_id}-{source_id}")
        if not base_id.startswith(f"{vehicle_id}-"):
            base_id = f"{vehicle_id}-{base_id}"
        record_id = base_id
        suffix = index
        while record_id in existing_ids:
            record_id = f"{base_id}-{suffix}"
            suffix += 1
        existing_ids.add(record_id)
        normalized.append(
            {
                "id": record_id,
                "vehicle_id": vehicle_id,
                "campaign_number": item.get("campaign_number", ""),
                "component": item.get("component") or "unknown",
                "summary": item.get("summary", ""),
                "consequence": item.get("consequence", ""),
                "remedy": item.get("remedy", ""),
                "source_id": source_id,
                "source_name": item.get("source_name") or item.get("source") or "NHTSA",
                "source_url": (
                    item.get("source_url")
                    or item.get("url")
                    or "https://www.nhtsa.gov/recalls"
                ),
                "captured_at": (
                    item.get("captured_at")
                    or item.get("observed_at")
                    or "unknown"
                ),
            }
        )
    updated = dict(catalog)
    updated["recalls"] = [
        item for item in catalog.get("recalls", []) if item.get("vehicle_id") != vehicle_id
    ] + normalized
    return updated


def _overlay_live_recalls_graph(
    graph: dict[str, list[dict[str, Any]]],
    vehicle_id: str,
    recalls: list[dict[str, Any]],
) -> None:
    old_recall_ids = {
        edge["end"]
        for edge in graph["edges"]
        if edge["start"] == vehicle_id and edge["relation"] == "AFFECTED_BY"
    }
    graph["edges"] = [
        edge
        for edge in graph["edges"]
        if not (
            (edge["start"] == vehicle_id and edge["relation"] == "AFFECTED_BY")
            or edge["start"] in old_recall_ids
            or edge["end"] in old_recall_ids
        )
    ]
    graph["nodes"] = [node for node in graph["nodes"] if node["id"] not in old_recall_ids]
    node_map = {node["id"]: node for node in graph["nodes"]}

    for recall in recalls:
        recall_id = recall["id"]
        props = {k: v for k, v in recall.items() if k not in {"id", "vehicle_id"}}
        node_map[recall_id] = {
            "label": "RecallCampaign",
            "id": recall_id,
            "properties": {"id": recall_id, **props},
        }
        graph["edges"].append(
            {
                "start": vehicle_id,
                "relation": "AFFECTED_BY",
                "end": recall_id,
                "properties": {"match_granularity": "make_model_year"},
            }
        )
        source_id = f"source-{recall['source_id']}"
        node_map[source_id] = {
            "label": "SourceDocument",
            "id": source_id,
            "properties": {
                "id": source_id,
                "source_id": recall["source_id"],
                "name": recall["source_name"],
                "url": recall["source_url"],
                "captured_at": recall["captured_at"],
            },
        }
        graph["edges"].append(
            {
                "start": recall_id,
                "relation": "WAS_DERIVED_FROM",
                "end": source_id,
                "properties": {},
            }
        )
        component = normalize_subsystem(recall.get("component", "unknown"))
        comp_id = component_node_id(component)
        node_map.setdefault(
            comp_id,
            {
                "label": "SubsystemComponent",
                "id": comp_id,
                "properties": {"id": comp_id, "name": component},
            },
        )
        graph["edges"].append(
            {
                "start": recall_id,
                "relation": "AFFECTS_COMPONENT",
                "end": comp_id,
                "properties": {},
            }
        )
    graph["nodes"] = list(node_map.values())
_GRAPH_CACHE: dict[int, tuple[dict[str, Any], tuple[int, ...], dict[str, list[dict[str, Any]]]]] = {}


def _catalog_counts(catalog: dict[str, Any]) -> tuple[int, ...]:
    return tuple(
        len(catalog.get(key, ()))
        for key in ("vehicles", "listings", "safety_ratings", "recalls", "reviews", "parts")
    )


def _cached_graph(catalog: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    counts = _catalog_counts(catalog)
    cached = _GRAPH_CACHE.get(id(catalog))
    if cached is None or cached[0] is not catalog or cached[1] != counts:
        _GRAPH_CACHE.clear()
        cached = (catalog, counts, build_graph(catalog))
        _GRAPH_CACHE[id(catalog)] = cached
    return cached[2]


def _retrieve_from_graph(
    graph: dict[str, list[dict[str, Any]]],
    vehicle_id: str,
    max_paths: int,
    backend: str,
    aspects: list[str] | None = None,
) -> list[dict[str, Any]]:
    if max_paths <= 0:
        return []
    nodes = {node["id"]: node for node in graph["nodes"]}
    if vehicle_id not in nodes:
        return []
    outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in graph["edges"]:
        outgoing[edge["start"]].append(edge)

    direct_by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    component_sources: dict[str, dict[str, dict[str, list[str]]]] = defaultdict(
        lambda: defaultdict(lambda: {"evidence_ids": [], "node_ids": [], "relations": []})
    )

    for edge in outgoing.get(vehicle_id, []):
        definition = DIRECT_PATHS.get(edge["relation"])
        if definition is None:
            continue
        kind, label = definition
        expanded = (
            _review_paths(vehicle_id, edge, nodes, outgoing)
            if kind == "review"
            else [_expand_path(vehicle_id, edge, kind, label, nodes, outgoing)]
        )
        for path in expanded:
            direct_by_kind[kind].append(path)
            component = path.get("component")
            if component and component != "unknown":
                bucket = component_sources[component][kind]
                bucket["evidence_ids"].extend(path["evidence_ids"])
                bucket["node_ids"].extend(path["node_ids"])
                bucket["relations"].extend(path["relations"])

    alignments = _build_alignments(vehicle_id, component_sources)
    lineage_paths = _generation_lineage_paths(vehicle_id, nodes, outgoing)

    requested_aspects = {str(a).strip().lower() for a in (aspects or []) if str(a).strip()}
    selected_reviews = _select_contrastive_reviews(
        direct_by_kind["review"],
        requested_aspects,
        cap=4,
    )
    for kind in ("listing", "safety", "recall", "product"):
        direct_by_kind[kind].sort(
            key=lambda path: (
                path.get("component") in {None, "unknown"},
                path["label"],
                tuple(path.get("evidence_ids", [])),
            )
        )

    lineage_limit = min(len(lineage_paths), 2 if max_paths >= 4 else 1)
    alignment_limit = (
        min(len(alignments), max(1, max_paths // 4))
        if alignments and max_paths >= 3
        else 0
    )
    direct_budget = max(0, max_paths - lineage_limit - alignment_limit)

    selected_direct = _allocate_direct_paths(
        direct_by_kind,
        selected_reviews,
        direct_budget,
    )

    # Reclaim unused direct budget for extra alignments/lineage if space remains
    remaining = max(0, max_paths - len(selected_direct) - lineage_limit - alignment_limit)
    if remaining and len(alignments) > alignment_limit:
        extra_align = min(remaining, len(alignments) - alignment_limit)
        alignment_limit += extra_align

    selected = selected_direct + lineage_paths[:lineage_limit] + alignments[:alignment_limit]
    for path in selected:
        path["backend"] = backend
    return selected[:max_paths]


def _build_alignments(
    vehicle_id: str,
    component_sources: dict[str, dict[str, dict[str, list[str]]]],
) -> list[dict[str, Any]]:
    alignments: list[dict[str, Any]] = []
    for component, groups in sorted(component_sources.items()):
        if component == "unknown":
            continue
        available = {
            kind: info
            for kind, info in groups.items()
            if info["evidence_ids"] or info["node_ids"]
        }
        if len(available) < 2:
            continue
        kinds = sorted(
            available,
            key=lambda k: ({"recall": 0, "review": 1, "product": 2}.get(k, 9), k),
        )
        node_ids = _unique(
            [vehicle_id]
            + [nid for kind in kinds for nid in available[kind]["node_ids"]]
            + [component_node_id(component)]
        )
        relations = _unique(
            rel
            for kind in kinds
            for rel in available[kind]["relations"]
            if rel != "WAS_DERIVED_FROM"
        )
        evidence_ids = _unique(
            ev_id for kind in kinds for ev_id in available[kind]["evidence_ids"]
        )
        if not evidence_ids:
            continue
        alignments.append(
            {
                "kind": "cross_source_alignment",
                "label": f"Cross-source {component.replace('_', ' ')} evidence",
                "component": component,
                "source_kinds": kinds,
                "node_ids": node_ids,
                "relations": relations,
                "evidence_ids": evidence_ids,
                "text": (
                    f"{', '.join(kind.replace('_', ' ') for kind in kinds)} records "
                    f"align on the {component.replace('_', ' ')} subsystem "
                    f"({len(evidence_ids)} linked record(s)). This is related maintenance "
                    "context, not proof of causation, fitment, or repair cost."
                ),
                "indirect": False,
            }
        )
    alignments.sort(
        key=lambda item: (
            "recall" not in item["source_kinds"],
            -len(item["source_kinds"]),
            item["component"],
        )
    )
    return alignments


def _count_exact_year_assertions(
    vehicle_id: str,
    nodes: dict[str, dict[str, Any]],
    outgoing: dict[str, list[dict[str, Any]]],
) -> int:
    assertion_ids: set[str] = set()
    for edge in outgoing.get(vehicle_id, []):
        if edge["relation"] != "HAS_REVIEW":
            continue
        match_type = edge.get("properties", {}).get("match_type", "exact_year")
        if match_type != "exact_year":
            continue
        review_id = edge["end"]
        for ass_edge in outgoing.get(review_id, []):
            if ass_edge["relation"] == "HAS_ASSERTION" and ass_edge["end"] in nodes:
                assertion_ids.add(ass_edge["end"])
    return len(assertion_ids)


def _generation_lineage_paths(
    vehicle_id: str,
    nodes: dict[str, dict[str, Any]],
    outgoing: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    if _count_exact_year_assertions(vehicle_id, nodes, outgoing) >= EXACT_YEAR_LINAGE_THRESHOLD:
        return []

    candidates: list[dict[str, Any]] = []
    seen_review_assertion: set[tuple[str, str]] = set()
    for edge in outgoing.get(vehicle_id, []):
        if edge["relation"] != "SHARES_GENERATION":
            continue
        sibling_id = edge["end"]
        sibling_node = nodes.get(sibling_id)
        if sibling_node is None:
            continue
        sibling_props = sibling_node["properties"]
        sibling_label = (
            f"{sibling_props.get('year', '')} {sibling_props.get('make', '')} "
            f"{sibling_props.get('model', '')}"
        ).strip() or sibling_id
        for rev_edge in outgoing.get(sibling_id, []):
            if rev_edge["relation"] != "HAS_REVIEW":
                continue
            if rev_edge.get("properties", {}).get("match_type", "exact_year") != "exact_year":
                continue
            review_id = rev_edge["end"]
            source_id = _source_id(review_id, nodes, outgoing)
            for ass_edge in outgoing.get(review_id, []):
                if ass_edge["relation"] != "HAS_ASSERTION":
                    continue
                assertion_id = ass_edge["end"]
                assertion_node = nodes.get(assertion_id)
                if assertion_node is None:
                    continue
                pair = (review_id, assertion_id)
                if pair in seen_review_assertion:
                    continue
                seen_review_assertion.add(pair)
                props = assertion_node["properties"]
                aspect = str(props.get("aspect") or "review")
                polarity = str(props.get("polarity") or "neutral")
                span = _truncate_span(str(props.get("span") or ""))
                component = None
                for comp_edge in outgoing.get(assertion_id, []):
                    if comp_edge["relation"] == "MENTIONS_COMPONENT":
                        comp_node = nodes.get(comp_edge["end"])
                        if comp_node is not None:
                            component = str(comp_node["properties"].get("name", "unknown"))
                            break
                aspect_label = aspect.replace("_", " ")
                snippet = f': "{span}"' if span else ""
                candidates.append(
                    {
                        "kind": "generation_lineage",
                        "label": f"Same-generation {aspect_label} ({polarity})",
                        "component": component,
                        "aspect": aspect,
                        "polarity": polarity,
                        "span": span,
                        "node_ids": [vehicle_id, sibling_id, review_id, assertion_id],
                        "relations": [
                            "SHARES_GENERATION",
                            "HAS_REVIEW",
                            "HAS_ASSERTION",
                        ],
                        "evidence_ids": [source_id] if source_id else [],
                        "text": (
                            f"Indirect generation lineage via {sibling_label} "
                            f"({polarity} {aspect_label}{snippet}). "
                            "Excluded from ranking scores."
                        ),
                        "indirect": True,
                    }
                )

    candidates.sort(
        key=lambda item: (
            item["polarity"] not in {"positive", "negative"},
            item["aspect"],
            tuple(item["evidence_ids"]),
        )
    )
    selected: list[dict[str, Any]] = []
    seen_reviews: set[str] = set()
    for item in candidates:
        rev_id = item["node_ids"][2]
        if rev_id in seen_reviews:
            continue
        seen_reviews.add(rev_id)
        selected.append(item)
        if len(selected) == 2:
            return selected
    for item in candidates:
        if item not in selected:
            selected.append(item)
            if len(selected) == 2:
                break
    return selected


def _select_contrastive_reviews(
    review_paths: list[dict[str, Any]],
    requested_aspects: set[str],
    cap: int = 4,
) -> list[dict[str, Any]]:
    if not review_paths or cap <= 0:
        return []

    def sort_key(path: dict[str, Any]) -> tuple[Any, ...]:
        aspect = str(path.get("aspect") or "").lower()
        return (
            bool(path.get("indirect")),
            bool(requested_aspects) and aspect not in requested_aspects,
            path.get("component") in {None, "unknown"},
            path.get("polarity") == "mixed",
            aspect,
            tuple(path.get("evidence_ids", [])),
        )

    ordered = sorted(review_paths, key=sort_key)
    if requested_aspects:
        preferred = [
            p for p in ordered if str(p.get("aspect") or "").lower() in requested_aspects
        ]
        fallback = [
            p for p in ordered if str(p.get("aspect") or "").lower() not in requested_aspects
        ]
    else:
        preferred = ordered
        fallback = []

    selected: list[dict[str, Any]] = []
    used_ids: set[int] = set()

    def pick_balanced(pool: list[dict[str, Any]]) -> None:
        if not pool or len(selected) >= cap:
            return
        positives = [p for p in pool if p.get("polarity") == "positive" and id(p) not in used_ids]
        negatives = [
            p
            for p in pool
            if p.get("polarity") in {"negative", "mixed"} and id(p) not in used_ids
        ]
        others = [
            p
            for p in pool
            if p.get("polarity") not in {"positive", "negative", "mixed"}
            and id(p) not in used_ids
        ]
        covered_pos_aspects: set[str] = set()
        covered_neg_aspects: set[str] = set()

        def pop_next(bucket: list[dict[str, Any]], covered: set[str]) -> dict[str, Any] | None:
            for idx, item in enumerate(bucket):
                asp = str(item.get("aspect") or "")
                if asp not in covered:
                    covered.add(asp)
                    return bucket.pop(idx)
            return bucket.pop(0) if bucket else None

        turn = 0
        while len(selected) < cap and (positives or negatives):
            bucket, covered = (
                (positives, covered_pos_aspects)
                if turn % 2 == 0
                else (negatives, covered_neg_aspects)
            )
            chosen = pop_next(bucket, covered)
            if chosen is None:
                other_bucket, other_cov = (
                    (negatives, covered_neg_aspects)
                    if turn % 2 == 0
                    else (positives, covered_pos_aspects)
                )
                chosen = pop_next(other_bucket, other_cov)
            if chosen is None:
                break
            used_ids.add(id(chosen))
            selected.append(chosen)
            turn += 1

        for item in others:
            if len(selected) >= cap:
                break
            if id(item) not in used_ids:
                used_ids.add(id(item))
                selected.append(item)

    pick_balanced(preferred)
    if len(selected) < cap and fallback:
        pick_balanced(fallback)
    return selected


def _allocate_direct_paths(
    direct_by_kind: dict[str, list[dict[str, Any]]],
    selected_reviews: list[dict[str, Any]],
    direct_budget: int,
) -> list[dict[str, Any]]:
    if direct_budget <= 0:
        return []
    pools: dict[str, list[dict[str, Any]]] = {
        "listing": list(direct_by_kind["listing"]),
        "safety": list(direct_by_kind["safety"]),
        "recall": list(direct_by_kind["recall"]),
        "review": list(selected_reviews),
        "product": list(direct_by_kind["product"]),
    }
    taken: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def take(kind: str, count: int) -> None:
        while count > 0 and pools[kind] and total_taken() < direct_budget:
            taken[kind].append(pools[kind].pop(0))
            count -= 1

    def total_taken() -> int:
        return sum(len(items) for items in taken.values())

    # Pass 1: guarantee representation for every evidence modality + contrastive pair
    take("listing", 1)
    take("safety", 1)
    take("recall", 1)
    take("review", 2)
    take("product", 1)

    # Pass 2: fill remaining budget up to balanced per-kind caps
    extra_caps = (("review", 2), ("recall", 1), ("product", 1), ("listing", 1))
    for kind, extra in extra_caps:
        take(kind, extra)

    ordered: list[dict[str, Any]] = []
    for kind in ("listing", "safety", "recall", "review", "product"):
        ordered.extend(taken[kind])
    return ordered


def _neo4j_neighborhood(
    vehicle_id: str,
    aspects: list[str] | None = None,
    max_paths: int = 12,
) -> dict[str, Any]:
    try:
        from neo4j import GraphDatabase
    except ImportError as error:
        raise RuntimeError("Neo4j support requires the graph dependencies") from error
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "clickless-local")
    with GraphDatabase.driver(
        uri,
        auth=(user, password),
        connection_timeout=2.0,
    ) as driver:
        driver.verify_connectivity()
        with driver.session() as session:
            node_rows = list(
                session.run(
                    "MATCH p=(v:VehicleVariant {id: $vehicle_id})-[*0..3]->(n) "
                    "UNWIND nodes(p) AS item "
                    "RETURN DISTINCT labels(item) AS labels, properties(item) AS properties",
                    vehicle_id=vehicle_id,
                )
            )
            edge_rows = list(
                session.run(
                    "MATCH p=(v:VehicleVariant {id: $vehicle_id})-[*1..3]->(n) "
                    "UNWIND relationships(p) AS rel "
                    "RETURN DISTINCT startNode(rel).id AS start, type(rel) AS relation, "
                    "endNode(rel).id AS end, properties(rel) AS properties",
                    vehicle_id=vehicle_id,
                )
            )
            sibling_node_rows = list(
                session.run(
                    "MATCH p=(v:VehicleVariant {id: $vehicle_id})-[:SHARES_GENERATION]->"
                    "(sibling:VehicleVariant)-[:HAS_REVIEW]->(r:Review)-[:HAS_ASSERTION|WAS_DERIVED_FROM*1..2]->(leaf) "
                    "UNWIND nodes(p) AS item "
                    "RETURN DISTINCT labels(item) AS labels, properties(item) AS properties",
                    vehicle_id=vehicle_id,
                )
            )
            sibling_edge_rows = list(
                session.run(
                    "MATCH p=(v:VehicleVariant {id: $vehicle_id})-[:SHARES_GENERATION]->"
                    "(sibling:VehicleVariant)-[:HAS_REVIEW]->(r:Review)-[:HAS_ASSERTION|WAS_DERIVED_FROM*1..2]->(leaf) "
                    "UNWIND relationships(p) AS rel "
                    "RETURN DISTINCT startNode(rel).id AS start, type(rel) AS relation, "
                    "endNode(rel).id AS end, properties(rel) AS properties",
                    vehicle_id=vehicle_id,
                )
            )
        template_results = run_query_templates(
            driver,
            vehicle_id,
            list(aspects or []),
            limit=max_paths,
        )

    node_map: dict[str, dict[str, Any]] = {}
    for row in node_rows + sibling_node_rows:
        properties = dict(row["properties"])
        labels = [label for label in row["labels"] if label != "Entity"]
        node_id = properties["id"]
        node_map[node_id] = {
            "id": node_id,
            "label": labels[0] if labels else "Entity",
            "properties": properties,
        }
    edge_map: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in edge_rows + sibling_edge_rows:
        key = (row["start"], row["relation"], row["end"])
        edge_map[key] = {
            "start": row["start"],
            "relation": row["relation"],
            "end": row["end"],
            "properties": dict(row["properties"]),
        }
    if not node_map:
        raise RuntimeError(f"Vehicle {vehicle_id} is not loaded in Neo4j")
    return {
        "nodes": list(node_map.values()),
        "edges": list(edge_map.values()),
        "template_results": template_results,
    }


def _expand_path(
    vehicle_id: str,
    first_edge: dict[str, Any],
    kind: str,
    label: str,
    nodes: dict[str, dict[str, Any]],
    outgoing: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    entity_id = first_edge["end"]
    entity = nodes[entity_id]
    node_ids = [vehicle_id, entity_id]
    relations = [first_edge["relation"]]
    evidence_ids = []
    component = None

    for edge in outgoing.get(entity_id, []):
        target = nodes.get(edge["end"])
        if target is None:
            continue
        if edge["relation"] == "WAS_DERIVED_FROM":
            node_ids.append(edge["end"])
            relations.append(edge["relation"])
            source_id = target["properties"].get("source_id")
            if source_id:
                evidence_ids.append(str(source_id))
        elif edge["relation"] in {"AFFECTS_COMPONENT", "FOR_COMPONENT"}:
            node_ids.append(edge["end"])
            relations.append(edge["relation"])
            component = str(target["properties"].get("name", "unknown"))
        elif edge["relation"] == "HAS_ASSERTION":
            assertion_id = edge["end"]
            node_ids.append(assertion_id)
            relations.append(edge["relation"])
            for assertion_edge in outgoing.get(assertion_id, []):
                if assertion_edge["relation"] != "MENTIONS_COMPONENT":
                    continue
                target_component = nodes.get(assertion_edge["end"])
                if target_component is None:
                    continue
                node_ids.append(assertion_edge["end"])
                relations.append(assertion_edge["relation"])
                component = str(target_component["properties"].get("name", "unknown"))

    properties = entity["properties"]
    if kind == "review":
        source_id = _source_id(entity_id, nodes, outgoing)
        if source_id:
            evidence_ids.append(source_id)
    indirect = bool(
        kind == "review" and first_edge.get("properties", {}).get("match_type", "exact_year") != "exact_year"
    )
    return {
        "kind": kind,
        "label": label,
        "component": component,
        "node_ids": _unique(node_ids),
        "relations": relations,
        "evidence_ids": _unique(evidence_ids),
        "text": _path_text(kind, properties, component),
        "indirect": indirect,
    }


def _review_paths(
    vehicle_id: str,
    first_edge: dict[str, Any],
    nodes: dict[str, dict[str, Any]],
    outgoing: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    review_id = first_edge["end"]
    source_id = _source_id(review_id, nodes, outgoing)
    indirect = first_edge.get("properties", {}).get("match_type", "exact_year") != "exact_year"
    paths = []
    for edge in outgoing.get(review_id, []):
        if edge["relation"] != "HAS_ASSERTION":
            continue
        assertion_id = edge["end"]
        assertion = nodes.get(assertion_id)
        if assertion is None:
            continue
        props = assertion["properties"]
        aspect = str(props.get("aspect", "review"))
        polarity = str(props.get("polarity", "neutral"))
        span = _truncate_span(str(props.get("span") or ""))
        aspect_label = aspect.replace("_", " ")
        comp_edges = [
            item
            for item in outgoing.get(assertion_id, [])
            if item["relation"] == "MENTIONS_COMPONENT" and item["end"] in nodes
        ]
        if comp_edges:
            for component_edge in comp_edges:
                component_node = nodes[component_edge["end"]]
                component = str(component_node["properties"].get("name", "unknown"))
                component_label = component.replace("_", " ")
                snippet = f': "{span}"' if span else ""
                paths.append(
                    {
                        "kind": "review",
                        "label": f"Owner review: {aspect_label} ({polarity})",
                        "component": component,
                        "aspect": aspect,
                        "polarity": polarity,
                        "span": span,
                        "node_ids": [
                            vehicle_id,
                            review_id,
                            assertion_id,
                            component_edge["end"],
                        ],
                        "relations": [
                            first_edge["relation"],
                            edge["relation"],
                            component_edge["relation"],
                        ],
                        "evidence_ids": [source_id] if source_id else [],
                        "text": (
                            f"Owner review {polarity} {aspect_label} assertion "
                            f"about the {component_label} subsystem{snippet}."
                        ),
                        "indirect": indirect,
                    }
                )
        else:
            snippet = f': "{span}"' if span else ""
            paths.append(
                {
                    "kind": "review",
                    "label": f"Owner review: {aspect_label} ({polarity})",
                    "component": None,
                    "aspect": aspect,
                    "polarity": polarity,
                    "span": span,
                    "node_ids": [vehicle_id, review_id, assertion_id],
                    "relations": [first_edge["relation"], edge["relation"]],
                    "evidence_ids": [source_id] if source_id else [],
                    "text": f"Owner review {polarity} {aspect_label} assertion{snippet}.",
                    "indirect": indirect,
                }
            )
    if paths:
        return paths
    return [
        _expand_path(
            vehicle_id,
            first_edge,
            "review",
            "Owner review",
            nodes,
            outgoing,
        )
    ]


def _truncate_span(text: str, limit: int = SPAN_MAX_CHARS) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def _source_id(
    entity_id: str,
    nodes: dict[str, dict[str, Any]],
    outgoing: dict[str, list[dict[str, Any]]],
) -> str | None:
    for edge in outgoing.get(entity_id, []):
        if edge["relation"] != "WAS_DERIVED_FROM":
            continue
        source = nodes.get(edge["end"])
        if source:
            value = source["properties"].get("source_id")
            return str(value) if value else None
    return None


def _path_text(kind: str, properties: dict[str, Any], component: str | None) -> str:
    part = (component or "unknown").replace("_", " ")
    if kind == "listing":
        return f"Listed at ${properties['price']:,} with {properties['mileage']:,} miles."
    if kind == "safety":
        from clickless.data.nhtsa import safety_stars

        stars, basis = safety_stars(properties)
        if stars is None:
            return "NHTSA safety record (no crash-test scores)."
        return f"NHTSA crash tests: {stars} stars" + (" on average." if basis == "test_average" else " overall.")
    if kind == "recall":
        return "Recall" + (f" affecting the {part}." if part != "unknown" else ".")
    if kind == "review":
        return "Owner review" + (f" that mentions the {part}." if component and part != "unknown" else ".")
    fit = "fit confirmed" if properties.get("verified_fitment") else "fit not confirmed"
    return "Replacement part" + (f" for the {part}" if part != "unknown" else "") + f" ({fit})."


def _unique(values: Any) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))
