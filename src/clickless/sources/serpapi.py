from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import requests

from .base import ProductOffer, SourceError


class SerpApiClient:
    ENDPOINT = "https://serpapi.com/search.json"

    def __init__(self, api_key: str, timeout: float = 30.0) -> None:
        self.api_key = api_key
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "SerpApiClient | None":
        key = os.getenv("SERPAPI_API_KEY", "").strip()
        return cls(key) if key else None

    def search_products(self, query: str, limit: int = 5) -> list[ProductOffer]:
        try:
            response = requests.get(
                self.ENDPOINT,
                params={
                    "engine": "google_shopping",
                    "q": query,
                    "gl": "us",
                    "hl": "en",
                    "api_key": self.api_key,
                },
                timeout=self.timeout,
            )
        except requests.RequestException as error:
            raise SourceError(f"SerpApi request failed: {error}") from error
        if response.status_code >= 400:
            raise SourceError(f"SerpApi returned HTTP {response.status_code}")
        payload = response.json()
        if payload.get("error"):
            raise SourceError(f"SerpApi error: {payload['error']}")
        captured_at = datetime.now(timezone.utc).isoformat()
        offers = []
        for item in payload.get("shopping_results", [])[: max(limit, 1)]:
            parsed = _offer(item, captured_at)
            if parsed is not None:
                offers.append(parsed)
        return offers


def _offer(item: dict[str, Any], captured_at: str) -> ProductOffer | None:
    price = _float(item.get("extracted_price"))
    url = str(item.get("link") or item.get("product_link") or "").strip()
    identifier = str(item.get("product_id") or url).strip()
    title = str(item.get("title") or "").strip()
    if price is None or not all((url, identifier, title)):
        return None
    snippet = str(item.get("snippet") or "").strip()
    reviews = (snippet,) if snippet else ()
    return ProductOffer(
        source="SerpApi Google Shopping",
        identifier=identifier,
        title=title,
        price=price,
        url=url,
        merchant=str(item.get("source") or "Google Shopping"),
        rating=_float(item.get("rating")),
        review_count=_integer(item.get("reviews")),
        review_texts=reviews,
        captured_at=captured_at,
        attributes={
            "condition": item.get("second_hand_condition"),
            "delivery": item.get("delivery"),
            "multiple_sources": item.get("multiple_sources"),
        },
    )


def _float(value: object) -> float | None:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _integer(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None
