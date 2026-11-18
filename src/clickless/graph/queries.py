from __future__ import annotations

from typing import Any


CROSS_SOURCE_ALIGNMENT = """
MATCH (v:VehicleVariant {id: $vehicle_id})
MATCH (component:SubsystemComponent)
WHERE component.name <> 'unknown'
  AND (
    (v)-[:AFFECTED_BY]->(:RecallCampaign)-[:AFFECTS_COMPONENT]->(component)
    OR (v)-[:HAS_REVIEW]->(:Review)-[:HAS_ASSERTION]->(:AspectAssertion)-[:MENTIONS_COMPONENT]->(component)
    OR (v)-[:COMPATIBLE_WITH|RELATED_PRODUCT]->(:AftermarketProduct)-[:FOR_COMPONENT]->(component)
  )
OPTIONAL MATCH (v)-[:AFFECTED_BY]->(recall:RecallCampaign)-[:AFFECTS_COMPONENT]->(component)
OPTIONAL MATCH (v)-[:HAS_REVIEW]->(review:Review)-[:HAS_ASSERTION]->(assertion:AspectAssertion)-[:MENTIONS_COMPONENT]->(component)
OPTIONAL MATCH (v)-[:COMPATIBLE_WITH|RELATED_PRODUCT]->(product:AftermarketProduct)-[:FOR_COMPONENT]->(component)
WITH component,
     [item IN collect(DISTINCT recall) WHERE item IS NOT NULL] AS recalls,
     [item IN collect(DISTINCT review) WHERE item IS NOT NULL] AS reviews,
     [item IN collect(DISTINCT assertion) WHERE item IS NOT NULL] AS assertions,
     [item IN collect(DISTINCT product) WHERE item IS NOT NULL] AS products
WHERE (CASE WHEN size(recalls) > 0 THEN 1 ELSE 0 END)
    + (CASE WHEN size(assertions) > 0 THEN 1 ELSE 0 END)
    + (CASE WHEN size(products) > 0 THEN 1 ELSE 0 END) >= 2
RETURN recalls, component, reviews, assertions, products
LIMIT $limit
"""

GENERATION_LINEAGE = """
MATCH (v:VehicleVariant {id: $vehicle_id})
OPTIONAL MATCH (v)-[er:HAS_REVIEW]->(:Review)-[:HAS_ASSERTION]->(exact_assertion:AspectAssertion)
WHERE er.match_type = 'exact_year' OR er.match_granularity = 'exact_year'
WITH v, count(DISTINCT exact_assertion) AS exact_assertion_count
WHERE exact_assertion_count < 5
MATCH (v)-[:SHARES_GENERATION]->(sibling:VehicleVariant)-[sr:HAS_REVIEW]->(review:Review)-[:HAS_ASSERTION]->(assertion:AspectAssertion)
WHERE sibling.id <> v.id AND coalesce(sr.match_type, 'exact_year') = 'exact_year'
OPTIONAL MATCH (v)-[:MEMBER_OF]->(generation:Generation)
RETURN sibling, review, assertion, generation, exact_assertion_count
LIMIT $limit
"""

CONTRASTIVE_BUNDLE = """
MATCH (v:VehicleVariant {id: $vehicle_id})-[:HAS_REVIEW]->(review:Review)-[:HAS_ASSERTION]->(assertion:AspectAssertion)
WHERE (size(coalesce($aspects, [])) = 0 OR assertion.aspect IN $aspects)
  AND assertion.polarity IN ['positive', 'negative', 'mixed', 'neutral']
OPTIONAL MATCH (v)-[:HAS_RATING]->(rating:SafetyRating)
OPTIONAL MATCH (v)-[:COMPATIBLE_WITH|RELATED_PRODUCT]->(product:AftermarketProduct)
RETURN assertion.aspect AS aspect,
       assertion.polarity AS polarity,
       collect(DISTINCT assertion)[0..3] AS assertions,
       collect(DISTINCT review)[0..3] AS reviews,
       collect(DISTINCT rating) AS ratings,
       collect(DISTINCT product)[0..3] AS products
LIMIT $limit
"""


def run_query_templates(
    driver: Any,
    vehicle_id: str,
    aspects: list[str] | None = None,
    limit: int = 12,
) -> dict[str, list[dict[str, Any]]]:
    aspect_filter = list(aspects or [])
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
                    aspects=aspect_filter,
                    limit=limit,
                )
            ],
        }
