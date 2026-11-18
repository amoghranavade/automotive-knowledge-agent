from __future__ import annotations

from typing import Any


CROSS_SOURCE_ALIGNMENT = """
MATCH (v:VehicleVariant {id: $vehicle_id})-[:AFFECTED_BY]->(recall:RecallCampaign)
MATCH (recall)-[:AFFECTS_COMPONENT]->(component:SubsystemComponent)
OPTIONAL MATCH (v)-[:HAS_REVIEW]->(review:Review)-[:HAS_ASSERTION]->(assertion:AspectAssertion)-[:MENTIONS_COMPONENT]->(component)
OPTIONAL MATCH (v)-[:COMPATIBLE_WITH|RELATED_PRODUCT]->(product:AftermarketProduct)-[:FOR_COMPONENT]->(component)
RETURN recall, component, collect(DISTINCT review) AS reviews, collect(DISTINCT assertion) AS assertions, collect(DISTINCT product) AS products
LIMIT $limit
"""

GENERATION_LINEAGE = """
MATCH (v:VehicleVariant {id: $vehicle_id})-[:MEMBER_OF]->(generation:Generation)<-[:MEMBER_OF]-(sibling:VehicleVariant)
WHERE sibling.id <> v.id
MATCH (sibling)-[:HAS_REVIEW]->(review:Review)-[:HAS_ASSERTION]->(assertion:AspectAssertion)
RETURN sibling, review, assertion, generation
LIMIT $limit
"""

CONTRASTIVE_BUNDLE = """
MATCH (v:VehicleVariant {id: $vehicle_id})-[:HAS_REVIEW]->(review:Review)-[:HAS_ASSERTION]->(assertion:AspectAssertion)
WHERE assertion.aspect IN $aspects
OPTIONAL MATCH (v)-[:HAS_RATING]->(rating:SafetyRating)
OPTIONAL MATCH (v)-[:COMPATIBLE_WITH|RELATED_PRODUCT]->(product:AftermarketProduct)
RETURN assertion.aspect AS aspect, assertion.polarity AS polarity, collect(DISTINCT review)[0..3] AS reviews, collect(DISTINCT rating) AS ratings, collect(DISTINCT product)[0..3] AS products
LIMIT $limit
"""


def run_query_templates(
    driver: Any,
    vehicle_id: str,
    aspects: list[str],
    limit: int = 12,
) -> dict[str, list[dict[str, Any]]]:
    with driver.session() as session:
        return {
            "cross_source_alignment": [
                record.data()
                for record in session.run(
                    CROSS_SOURCE_ALIGNMENT, vehicle_id=vehicle_id, limit=limit
                )
            ],
            "generation_lineage": [
                record.data()
                for record in session.run(
                    GENERATION_LINEAGE, vehicle_id=vehicle_id, limit=limit
                )
            ],
            "contrastive_bundle": [
                record.data()
                for record in session.run(
                    CONTRASTIVE_BUNDLE,
                    vehicle_id=vehicle_id,
                    aspects=aspects,
                    limit=limit,
                )
            ],
        }
