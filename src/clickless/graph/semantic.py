from __future__ import annotations

from pathlib import Path
from typing import Any

from rdflib import Graph, Literal, Namespace, RDF, URIRef
from rdflib.namespace import PROV, SH, XSD

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


def catalog_to_rdf(catalog: dict[str, Any]) -> Graph:
    graph = Graph()
    graph.bind("clickless", CLICKLESS)
    graph.bind("schema", SCHEMA)
    graph.bind("prov", PROV)
    built = build_graph(catalog)
    for node in built["nodes"]:
        subject = CLICKLESS[node["id"]]
        graph.add((subject, RDF.type, LABEL_CLASS.get(node["label"], CLICKLESS[node["label"]])))
        for key, value in node["properties"].items():
            if value is None or isinstance(value, (list, dict)):
                continue
            graph.add((subject, CLICKLESS[key], _literal(value)))
            standard_property = SCHEMA_PROPERTIES.get((node["label"], key))
            if standard_property is not None:
                graph.add(
                    (
                        subject,
                        standard_property,
                        URIRef(str(value)) if key in {"url", "source_url"} else _literal(value),
                    )
                )
    for edge in built["edges"]:
        predicate = PROV.wasDerivedFrom if edge["relation"] == "WAS_DERIVED_FROM" else CLICKLESS[edge["relation"]]
        graph.add((CLICKLESS[edge["start"]], predicate, CLICKLESS[edge["end"]]))
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
    if str(value).startswith("http"):
        return Literal(str(value), datatype=XSD.anyURI)
    return Literal(str(value))
