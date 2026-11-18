from __future__ import annotations

from pathlib import Path

from clickless.crawlers.extract import extract_products
from clickless.graph import build_graph, retrieve_evidence_paths, validate_rdf, write_rdf


def test_build_graph_match_granularity_properties(mini_catalog) -> None:
    graph = build_graph(mini_catalog)
    edges_by_rel: dict[str, list[dict]] = {}
    for edge in graph["edges"]:
        edges_by_rel.setdefault(edge["relation"], []).append(edge)

    assert all(
        edge["properties"].get("match_granularity") in {"exact_year", "same_generation", "product_fitment_unverified"}
        for edge in edges_by_rel["HAS_REVIEW"]
    )
    assert any(
        edge["properties"].get("match_granularity") == "exact_year"
        for edge in edges_by_rel["HAS_REVIEW"]
    )
    assert all(
        "match_granularity" in edge["properties"]
        for edge in edges_by_rel["REVIEWS"]
    )
    assert all(
        edge["properties"].get("match_granularity") == "make_model_year"
        for edge in edges_by_rel["AFFECTED_BY"]
    )
    assert all(
        edge["properties"].get("match_granularity") == "variant_tested"
        for edge in edges_by_rel["HAS_RATING"]
    )
    assert all(
        edge["properties"].get("match_granularity") == "verified_fitment"
        for edge in edges_by_rel["COMPATIBLE_WITH"]
    )
    assert all(
        edge["properties"].get("match_granularity") == "make_model_subsystem"
        for edge in edges_by_rel["RELATED_PRODUCT"]
    )


def test_graphrag_three_patterns_and_live_recalls_overlay(mini_catalog) -> None:
    # Backward-compatible 2-arg signature
    paths_2009 = retrieve_evidence_paths(mini_catalog, "subaru-forester-2009-awd")
    assert paths_2009

    # Pattern 1: cross_source_alignment on brakes (recall + review + part)
    alignments = [p for p in paths_2009 if p["kind"] == "cross_source_alignment"]
    assert any(p["component"] == "brakes" for p in alignments)
    brake_alignment = next(p for p in alignments if p["component"] == "brakes")
    assert set(brake_alignment["source_kinds"]) >= {"recall", "review", "product"}
    assert "AFFECTED_BY" in brake_alignment["relations"]
    assert "AFFECTS_COMPONENT" in brake_alignment["relations"]
    assert "nhtsa-11V123000" in brake_alignment["evidence_ids"]
    assert "opinrank-forester-2009-0" in brake_alignment["evidence_ids"]
    assert "amazon-meta-forester-brakes" in brake_alignment["evidence_ids"]

    # Pattern 2: < 5 exact-year assertions gate for generation_lineage
    assert not any(p["kind"] == "generation_lineage" for p in paths_2009)

    paths_2010 = retrieve_evidence_paths(mini_catalog, "subaru-forester-2010-awd")
    lineage_2010 = [p for p in paths_2010 if p["kind"] == "generation_lineage"]
    assert 1 <= len(lineage_2010) <= 2
    assert all(p["indirect"] is True for p in lineage_2010)
    assert all("SHARES_GENERATION" in p["relations"] for p in lineage_2010)

    # Pattern 3: contrastive_bundle balances positive and negative/mixed polarities
    contrastive = retrieve_evidence_paths(
        mini_catalog,
        "subaru-forester-2009-awd",
        max_paths=12,
        aspects=["reliability", "maintenance", "fuel_economy", "cargo"],
    )
    review_paths = [p for p in contrastive if p["kind"] == "review"]
    assert review_paths
    for rpath in review_paths:
        assert "aspect" in rpath
        assert rpath["polarity"] in {"positive", "negative", "neutral", "mixed"}
        assert rpath.get("span")
        assert len(rpath["span"]) <= 140
    polarities = {r["polarity"] for r in review_paths}
    assert "positive" in polarities
    assert polarities & {"negative", "mixed"}

    # live_recalls overlays without mutating caller's catalog
    original_count = len(mini_catalog["recalls"])
    overlaid = retrieve_evidence_paths(
        mini_catalog,
        "subaru-forester-2010-awd",
        live_recalls=[
            {
                "id": "subaru-forester-2010-awd-nhtsa-20V999000",
                "vehicle_id": "subaru-forester-2010-awd",
                "campaign_number": "20V999000",
                "component": "SERVICE BRAKES",
                "summary": "Live recall notice on hydraulic brakes.",
                "source_id": "nhtsa-20V999000",
                "source_name": "NHTSA",
                "source_url": "https://www.nhtsa.gov/recalls",
                "captured_at": "2026-02-01T00:00:00Z",
            }
        ],
    )
    assert len(mini_catalog["recalls"]) == original_count
    assert any(
        p["kind"] == "recall" and "nhtsa-20V999000" in p["evidence_ids"]
        for p in overlaid
    )


def test_rdf_iri_escaping_and_shacl_validation(
    mini_catalog, tmp_path: Path
) -> None:
    shapes_path = Path(__file__).resolve().parents[1] / "data" / "shapes.ttl"
    ttl_path = tmp_path / "mini_catalog.ttl"
    write_rdf(mini_catalog, ttl_path)
    ttl_text = ttl_path.read_text(encoding="utf-8")
    assert "ebay-listing-v1%7C123456%7C0" in ttl_text

    conforms, report = validate_rdf(ttl_path, shapes_path)
    assert conforms is True, report

    # Negative SHACL check: invalid price (0) should violate OfferShape
    bad_catalog = {**mini_catalog, "listings": [dict(mini_catalog["listings"][0], price=0)]}
    bad_ttl = tmp_path / "bad_catalog.ttl"
    write_rdf(bad_catalog, bad_ttl)
    bad_conforms, _ = validate_rdf(bad_ttl, shapes_path)
    assert bad_conforms is False


def test_extract_products_jsonld_and_dom_css_fallbacks() -> None:
    captured_at = "2026-02-01T12:00:00Z"

    jsonld_html = """
    <html><head>
    <script type="application/ld+json">
    {
      "@context": "https://schema.org",
      "@type": "Product",
      "name": "Bosch QuietCast Brake Pads",
      "url": "https://www.amazon.com/dp/B00BOSCH1",
      "offers": {"@type": "Offer", "price": "54.99", "priceCurrency": "USD"},
      "aggregateRating": {"@type": "AggregateRating", "ratingValue": "4.8"}
    }
    </script></head><body></body></html>
    """
    jsonld_items = extract_products(jsonld_html, "amazon", captured_at)
    assert len(jsonld_items) == 1
    assert jsonld_items[0]["selector_version"] == "jsonld-v1"
    assert jsonld_items[0]["price"] == 54.99

    ebay_html = """
    <ul>
      <li class="s-item">
        <a class="s-item__link" href="/itm/98765">
          <div class="s-item__title">2009 Subaru Forester AWD</div>
        </a>
        <span class="s-item__price">$11,250.00</span>
      </li>
    </ul>
    """
    ebay_items = extract_products(ebay_html, "ebay", captured_at)
    assert len(ebay_items) == 1
    assert ebay_items[0]["selector_version"] == "dom-css-v2"
    assert ebay_items[0]["url"].startswith("https://www.ebay.com/")
    assert ebay_items[0]["price"] == 11250.0

    amazon_html = """
    <div data-component-type="s-search-result">
      <h2><a class="a-link-normal" href="/dp/B09BRAKE01"><span>Akebono Ceramic Pads</span></a></h2>
      <span class="a-price"><span class="a-offscreen">$62.45</span></span>
    </div>
    """
    amazon_items = extract_products(amazon_html, "amazon", captured_at)
    assert len(amazon_items) == 1
    assert amazon_items[0]["selector_version"] == "dom-css-v2"
    assert amazon_items[0]["url"].startswith("https://www.amazon.com/")
    assert amazon_items[0]["price"] == 62.45

    walmart_html = """
    <div data-item-id="WM12345">
      <a href="/ip/EverStart-AGM-Battery/12345">
        <span data-automation-id="product-title">EverStart AGM Battery</span>
      </a>
      <div data-automation-id="product-price">$179.97</div>
    </div>
    """
    walmart_items = extract_products(walmart_html, "walmart", captured_at)
    assert len(walmart_items) == 1
    assert walmart_items[0]["selector_version"] == "dom-css-v2"
    assert walmart_items[0]["url"].startswith("https://www.walmart.com/")
    assert walmart_items[0]["price"] == 179.97

    bestbuy_html = """
    <li class="sku-item">
      <h4 class="sku-title"><a href="/site/dash-cam/654321.p">Vantrue Dual Dash Cam</a></h4>
      <div class="priceView-customer-price"><span>$199.99</span></div>
    </li>
    """
    bestbuy_items = extract_products(bestbuy_html, "bestbuy", captured_at)
    assert len(bestbuy_items) == 1
    assert bestbuy_items[0]["selector_version"] == "dom-css-v2"
    assert bestbuy_items[0]["url"].startswith("https://www.bestbuy.com/")
    assert bestbuy_items[0]["price"] == 199.99
