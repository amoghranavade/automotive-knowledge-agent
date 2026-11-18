from __future__ import annotations

from typing import Any

from .build import build_graph


def load_into_neo4j(driver: Any, catalog: dict[str, Any], clear: bool = False) -> dict[str, int]:
    graph = build_graph(catalog)
    with driver.session() as session:
        if clear:
            session.run("MATCH (n) DETACH DELETE n")
        session.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE")
        for node in graph["nodes"]:
            label = _safe_token(node["label"])
            session.run(
                f"MERGE (n:Entity:{label} {{id: $id}}) SET n += $properties",
                id=node["id"],
                properties=node["properties"],
            )
        for edge in graph["edges"]:
            relation = _safe_token(edge["relation"])
            session.run(
                f"MATCH (a:Entity {{id: $start}}), (b:Entity {{id: $end}}) "
                f"MERGE (a)-[r:{relation}]->(b) SET r += $properties",
                start=edge["start"],
                end=edge["end"],
                properties=edge["properties"],
            )
    return {"nodes": len(graph["nodes"]), "edges": len(graph["edges"])}


def _safe_token(value: str) -> str:
    if not value.replace("_", "").isalnum():
        raise ValueError(f"Invalid graph token: {value}")
    return value
