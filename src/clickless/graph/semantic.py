from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import quote

from rdflib import Graph, Literal, Namespace, RDF, URIRef
from rdflib.namespace import PROV, XSD

from .build import build_graph


CLICKLESS = Namespace("https://clickless.example.org/resource/")
SCHEMA = Namespace("https://schema.org/")


LABEL_CLASS = {
    "VehicleVariant": SCHEMA.Vehicle,
    "Listing": SCHEMA.Offer,
    "Review": SCHEMA.Review,
    "AftermarketProduct": SCHEMA.Product,
    "SafetyRating": CLICKLESS.SafetyRating,
    "RecallCampaign": CLICKLESS.RecallCampaign,
    "AspectAssertion": CLICKLESS.AspectAssertion,
    "SubsystemComponent": CLICKLESS.SubsystemComponent,
    "Generation": CLICKLESS.Generation,
    "SourceDocument": PROV.Entity,
}

SCHEMA_PROPERTIES = {
    ("VehicleVariant", "year"): SCHEMA.vehicleModelDate,
    ("VehicleVariant", "make"): SCHEMA.manufacturer,
    ("VehicleVariant", "model"): SCHEMA.model,
    ("VehicleVariant", "body_type"): SCHEMA.bodyType,
    ("VehicleVariant", "drivetrain"): SCHEMA.driveWheelConfiguration,
    ("Listing", "price"): SCHEMA.price,
    ("Listing", "source_url"): SCHEMA.url,
    ("Review", "text"): SCHEMA.reviewBody,
    ("AftermarketProduct", "name"): SCHEMA.name,
    ("AftermarketProduct", "source_url"): SCHEMA.url,
    ("SourceDocument", "url"): SCHEMA.url,
    ("SourceDocument", "captured_at"): PROV.generatedAtTime,
}

GRANULARITY_RELATIONS = {
    "HAS_REVIEW",
    "AFFECTED_BY",
    "HAS_RATING",
    "COMPATIBLE_WITH",
    "RELATED_PRODUCT",
}


def _safe_uri(namespace: Namespace, raw_id: Any) -> URIRef:
    return namespace[quote(str(raw_id), safe="-._~")]


def _safe_url_ref(value: Any) -> URIRef:
    return URIRef(quote(str(value).strip(), safe=":/?&=#%-._~"))


def catalog_to_rdf(catalog: dict[str, Any]) -> Graph:
    graph = Graph()
    graph.bind("clickless", CLICKLESS)
    graph.bind("schema", SCHEMA)
    graph.bind("prov", PROV)
    built = build_graph(catalog)
    for node in built["nodes"]:
        subject = _safe_uri(CLICKLESS, node["id"])
        rdf_class = LABEL_CLASS.get(node["label"]) or _safe_uri(CLICKLESS, node["label"])
        graph.add((subject, RDF.type, rdf_class))
        for key, value in node["properties"].items():
            if value is None or isinstance(value, (list, dict)):
                continue
            graph.add((subject, _safe_uri(CLICKLESS, key), _literal(value)))
            standard_property = SCHEMA_PROPERTIES.get((node["label"], key))
            if standard_property is not None:
                graph.add(
                    (
                        subject,
                        standard_property,
                        _safe_url_ref(value) if key in {"url", "source_url"} else _literal(value),
                    )
                )
    for edge in built["edges"]:
        start_uri = _safe_uri(CLICKLESS, edge["start"])
        end_uri = _safe_uri(CLICKLESS, edge["end"])
        predicate = (
            PROV.wasDerivedFrom
            if edge["relation"] == "WAS_DERIVED_FROM"
            else _safe_uri(CLICKLESS, edge["relation"])
        )
        graph.add((start_uri, predicate, end_uri))
        granularity = (edge.get("properties") or {}).get("match_granularity")
        if granularity and edge["relation"] in GRANULARITY_RELATIONS:
            graph.add((end_uri, CLICKLESS.match_granularity, _literal(granularity)))
    return graph


def write_rdf(catalog: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    catalog_to_rdf(catalog).serialize(destination=destination, format="turtle")


def validate_rdf(data_path: Path, shapes_path: Path) -> tuple[bool, str]:
    try:
        from pyshacl import validate
    except ImportError as error:
        raise RuntimeError("Install the graph dependencies to run SHACL validation") from error
    conforms, _, report = validate(
        data_graph=str(data_path),
        shacl_graph=str(shapes_path),
        inference="rdfs",
        abort_on_first=False,
    )
    return bool(conforms), str(report)


def _literal(value: Any) -> Literal:
    if isinstance(value, bool):
        return Literal(value, datatype=XSD.boolean)
    if isinstance(value, int):
        return Literal(value, datatype=XSD.integer)
    if isinstance(value, float):
        return Literal(value, datatype=XSD.decimal)
    text = str(value)
    if text.startswith(("http://", "https://")):
        return Literal(quote(text.strip(), safe=":/?&=#%-._~"), datatype=XSD.anyURI)
    return Literal(text)
