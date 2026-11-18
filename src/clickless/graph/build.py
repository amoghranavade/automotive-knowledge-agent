from __future__ import annotations

from typing import Any
from urllib.parse import quote

from clickless.data.aspects import normalize_subsystem


def build_graph(catalog: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[tuple[str, str, str], dict[str, Any]] = {}

    def add_node(label: str, node_id: str, properties: dict[str, Any]) -> None:
        nodes[node_id] = {"label": label, "id": node_id, "properties": {"id": node_id, **properties}}

    def add_edge(start: str, relation: str, end: str, **properties: Any) -> None:
        edges[(start, relation, end)] = {
            "start": start,
            "relation": relation,
            "end": end,
            "properties": properties,
        }

    generation_members: dict[str, list[str]] = {}
    for vehicle in catalog["vehicles"]:
        generation_id = vehicle["generation_id"]
        generation_members.setdefault(generation_id, []).append(vehicle["id"])
        add_node("Generation", generation_id, {})
        add_node("VehicleVariant", vehicle["id"], _without(vehicle, "id"))
        add_edge(vehicle["id"], "MEMBER_OF", generation_id)
    for members in generation_members.values():
        for start in members:
            for end in members:
                if start != end:
                    add_edge(start, "SHARES_GENERATION", end)

    for collection, label, relation in (
        ("listings", "Listing", "HAS_LISTING"),
        ("safety_ratings", "SafetyRating", "HAS_RATING"),
        ("recalls", "RecallCampaign", "AFFECTED_BY"),
    ):
        for record in catalog[collection]:
            add_node(label, record["id"], _without(record, "id", "vehicle_id"))
            edge_props: dict[str, Any] = {}
            if relation == "HAS_RATING":
                edge_props["match_granularity"] = "variant_tested"
            elif relation == "AFFECTED_BY":
                edge_props["match_granularity"] = "make_model_year"
            add_edge(record["vehicle_id"], relation, record["id"], **edge_props)
            if collection == "listings":
                add_edge(record["id"], "LISTING_OF", record["vehicle_id"])
            _add_provenance(record, add_node, add_edge)

    for review in catalog["reviews"]:
        add_node("Review", review["id"], _without(review, "id", "vehicle_id", "aspects"))
        match_type = str(review.get("match_type") or "exact_year")
        review_granularity = (
            "exact_year"
            if match_type == "exact_year"
            else ("same_generation" if match_type == "same_generation" else match_type)
        )
        add_edge(
            review["vehicle_id"],
            "HAS_REVIEW",
            review["id"],
            match_type=match_type,
            match_granularity=review_granularity,
        )
        add_edge(
            review["id"],
            "REVIEWS",
            review["vehicle_id"],
            match_type=match_type,
            match_granularity=review_granularity,
        )
        _add_provenance(review, add_node, add_edge)
        for aspect in review["aspects"]:
            add_node("AspectAssertion", aspect["id"], _without(aspect, "id"))
            add_edge(review["id"], "HAS_ASSERTION", aspect["id"])
            component_id = component_node_id(aspect["subsystem"])
            add_node("SubsystemComponent", component_id, {"name": aspect["subsystem"]})
            add_edge(aspect["id"], "MENTIONS_COMPONENT", component_id)

    for recall in catalog["recalls"]:
        component = normalize_subsystem(recall["component"])
        component_id = component_node_id(component)
        add_node("SubsystemComponent", component_id, {"name": component})
        add_edge(recall["id"], "AFFECTS_COMPONENT", component_id)

    for part in catalog["parts"]:
        add_node("AftermarketProduct", part["id"], _without(part, "id", "vehicle_id"))
        verified = bool(part.get("verified_fitment"))
        relation = "COMPATIBLE_WITH" if verified else "RELATED_PRODUCT"
        part_granularity = "verified_fitment" if verified else "make_model_subsystem"
        add_edge(
            part["vehicle_id"],
            relation,
            part["id"],
            match_granularity=part_granularity,
        )
        component_id = component_node_id(part["subsystem"])
        add_node("SubsystemComponent", component_id, {"name": part["subsystem"]})
        add_edge(part["id"], "FOR_COMPONENT", component_id)
        _add_provenance(part, add_node, add_edge)

    return {"nodes": list(nodes.values()), "edges": list(edges.values())}


def _add_provenance(record: dict[str, Any], add_node: Any, add_edge: Any) -> None:
    source_id = f"source-{record['source_id']}"
    add_node(
        "SourceDocument",
        source_id,
        {
            "source_id": record["source_id"],
            "name": record["source_name"],
            "url": record["source_url"],
            "captured_at": record["captured_at"],
        },
    )
    add_edge(record["id"], "WAS_DERIVED_FROM", source_id)


def _without(record: dict[str, Any], *keys: str) -> dict[str, Any]:
    excluded = set(keys)
    return {key: value for key, value in record.items() if key not in excluded}


def component_node_id(value: Any) -> str:
    return f"component-{quote(str(value), safe='-._~')}"
