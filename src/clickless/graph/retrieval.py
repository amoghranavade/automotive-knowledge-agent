from __future__ import annotations

from collections import defaultdict
import os
from typing import Any

from .build import build_graph, component_node_id


DIRECT_PATHS = {
    "HAS_LISTING": ("listing", "Listing and source"),
    "HAS_RATING": ("safety", "Safety rating and source"),
    "AFFECTED_BY": ("recall", "Recall and affected component"),
    "HAS_REVIEW": ("review", "Owner review and aspect assertion"),
    "COMPATIBLE_WITH": ("product", "Compatible product and component"),
    "RELATED_PRODUCT": ("product", "Related product and component"),
}


def retrieve_evidence_paths(
    catalog: dict[str, Any],
    vehicle_id: str,
    max_paths: int = 12,
) -> list[dict[str, Any]]:
    backend = os.getenv("CLICKLESS_GRAPH_BACKEND", "memory").strip().lower()
    graph = build_graph(catalog)
    used_backend = "memory"
    if backend == "neo4j":
        try:
            graph = _neo4j_neighborhood(vehicle_id)
            used_backend = "neo4j"
        except Exception:
            graph = build_graph(catalog)
            used_backend = "memory_fallback"
    return _retrieve_from_graph(graph, vehicle_id, max_paths, used_backend)


def _retrieve_from_graph(
    graph: dict[str, list[dict[str, Any]]],
    vehicle_id: str,
    max_paths: int,
    backend: str,
) -> list[dict[str, Any]]:
    nodes = {node["id"]: node for node in graph["nodes"]}
    outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in graph["edges"]:
        outgoing[edge["start"]].append(edge)

    paths = []
    component_sources: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
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
            paths.append(path)
            component = path.get("component")
            if component:
                component_sources[component][kind].extend(path["evidence_ids"])

    alignments = []
    for component, groups in sorted(component_sources.items()):
        if component == "unknown":
            continue
        available = {kind: _unique(ids) for kind, ids in groups.items() if ids}
        if len(available) < 2:
            continue
        kinds = sorted(available)
        alignments.append(
            {
                "kind": "cross_source_alignment",
                "label": f"Cross-source {component.replace('_', ' ')} evidence",
                "component": component,
                "source_kinds": kinds,
                "node_ids": [vehicle_id, component_node_id(component)],
                "relations": ["SHARED_COMPONENT"],
                "evidence_ids": _unique(
                    evidence_id for kind in kinds for evidence_id in available[kind]
                ),
                "text": (
                    f"{', '.join(kind.replace('_', ' ') for kind in kinds)} records "
                    f"refer to the {component.replace('_', ' ')} subsystem. This is "
                    "related evidence, not proof of causation, fitment, or repair cost."
                ),
                "indirect": False,
            }
        )

    priority = {"listing": 0, "safety": 1, "recall": 2, "review": 3, "product": 4}
    paths.sort(
        key=lambda path: (
            priority.get(path["kind"], 9),
            path["kind"] == "review" and path.get("component") is None,
            path["label"],
        )
    )
    alignment_limit = min(len(alignments), max(1, max_paths // 3))
    path_limit = max(0, max_paths - alignment_limit)
    caps = {"listing": 2, "safety": 1, "recall": 2, "review": 3, "product": 2}
    counts: dict[str, int] = defaultdict(int)
    diverse = []
    for path in paths:
        kind = path["kind"]
        if counts[kind] >= caps.get(kind, 1):
            continue
        diverse.append(path)
        counts[kind] += 1
        if len(diverse) == path_limit:
            break
    selected = diverse + alignments[:alignment_limit]
    for path in selected:
        path["backend"] = backend
    return selected


def _neo4j_neighborhood(vehicle_id: str) -> dict[str, list[dict[str, Any]]]:
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
            node_rows = session.run(
                "MATCH p=(v:VehicleVariant {id: $vehicle_id})-[*0..3]->(n) "
                "UNWIND nodes(p) AS item "
                "RETURN DISTINCT labels(item) AS labels, properties(item) AS properties",
                vehicle_id=vehicle_id,
            )
            nodes = []
            for row in node_rows:
                properties = dict(row["properties"])
                labels = [label for label in row["labels"] if label != "Entity"]
                nodes.append(
                    {
                        "id": properties["id"],
                        "label": labels[0] if labels else "Entity",
                        "properties": properties,
                    }
                )
            edge_rows = session.run(
                "MATCH p=(v:VehicleVariant {id: $vehicle_id})-[*1..3]->(n) "
                "UNWIND relationships(p) AS rel "
                "RETURN DISTINCT startNode(rel).id AS start, type(rel) AS relation, "
                "endNode(rel).id AS end, properties(rel) AS properties",
                vehicle_id=vehicle_id,
            )
            edges = [
                {
                    "start": row["start"],
                    "relation": row["relation"],
                    "end": row["end"],
                    "properties": dict(row["properties"]),
                }
                for row in edge_rows
            ]
    if not nodes:
        raise RuntimeError(f"Vehicle {vehicle_id} is not loaded in Neo4j")
    return {"nodes": nodes, "edges": edges}


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
        kind == "review" and first_edge.get("properties", {}).get("match_type") != "exact_year"
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
    indirect = first_edge.get("properties", {}).get("match_type") != "exact_year"
    paths = []
    for edge in outgoing.get(review_id, []):
        if edge["relation"] != "HAS_ASSERTION":
            continue
        assertion_id = edge["end"]
        assertion = nodes.get(assertion_id)
        if assertion is None:
            continue
        for component_edge in outgoing.get(assertion_id, []):
            if component_edge["relation"] != "MENTIONS_COMPONENT":
                continue
            component_node = nodes.get(component_edge["end"])
            if component_node is None:
                continue
            component = str(component_node["properties"].get("name", "unknown"))
            aspect = str(assertion["properties"].get("aspect", "review"))
            paths.append(
                {
                    "kind": "review",
                    "label": f"Owner review: {aspect.replace('_', ' ')}",
                    "component": component,
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
                        f"Owner review contains a {aspect.replace('_', ' ')} assertion "
                        f"about the {component.replace('_', ' ')} subsystem."
                    ),
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
    if kind == "listing":
        return f"Listing at ${properties['price']:,} with {properties['mileage']:,} miles."
    if kind == "safety":
        return f"Safety record with {properties.get('overall_stars')} overall stars."
    if kind == "recall":
        return f"Recall record linked to the {(component or 'unknown').replace('_', ' ')} subsystem."
    if kind == "review":
        suffix = f" and the {component.replace('_', ' ')} subsystem" if component else ""
        return f"Owner review linked to an aspect assertion{suffix}."
    fitment = "verified fitment" if properties.get("verified_fitment") else "unverified fitment"
    return f"Aftermarket product linked to the {(component or 'unknown').replace('_', ' ')} subsystem; {fitment}."


def _unique(values: Any) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))
