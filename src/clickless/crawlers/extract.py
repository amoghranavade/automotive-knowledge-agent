from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Iterable


def extract_products(html: str, site: str, captured_at: str) -> list[dict[str, Any]]:
    try:
        from parsel import Selector
    except ImportError as error:
        raise RuntimeError("Scrapy/Parsel is required for retail extraction") from error
    selector = Selector(text=html)
    products = []
    for raw in selector.css('script[type="application/ld+json"]::text').getall():
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        for item in _walk(payload):
            if item.get("@type") not in {"Product", "Vehicle"}:
                continue
            offer = item.get("offers") or {}
            if isinstance(offer, list):
                offer = offer[0] if offer else {}
            products.append(
                {
                    "name": item.get("name"),
                    "url": item.get("url") or offer.get("url"),
                    "price": _float(offer.get("price") or offer.get("lowPrice")),
                    "currency": offer.get("priceCurrency"),
                    "rating": (item.get("aggregateRating") or {}).get("ratingValue"),
                    "source": site,
                    "captured_at": captured_at,
                    "selector_version": "jsonld-v1",
                    "data_mode": "live_capture",
                }
            )
    unique = {}
    for product in products:
        key = product.get("url") or f"{product.get('name')}:{product.get('price')}"
        unique[str(key)] = product
    return list(unique.values())


def _walk(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
