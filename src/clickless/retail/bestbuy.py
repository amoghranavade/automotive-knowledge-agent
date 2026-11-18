from __future__ import annotations

import os

import requests

import clickless.config  # noqa: F401

from .base import RetailError, RetailResult


class BestBuyClient:
    ENDPOINT = "https://api.bestbuy.com/v1/products"

    def __init__(self, api_key: str, timeout: float = 15.0) -> None:
        self.api_key = api_key
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "BestBuyClient | None":
        key = os.getenv("BESTBUY_API_KEY", "").strip()
        return cls(key) if key else None

    def search(self, query: str, limit: int = 20) -> list[RetailResult]:
        terms = "&".join(f"search={word}" for word in query.split() if word)
        response = requests.get(
            f"{self.ENDPOINT}({terms})",
            params={
                "apiKey": self.api_key,
                "format": "json",
                "pageSize": min(max(limit, 1), 100),
                "show": "sku,name,salePrice,url,image,manufacturer,customerReviewAverage",
            },
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            raise RetailError(f"Best Buy search returned HTTP {response.status_code}")
        results = []
        for product in response.json().get("products", []):
            results.append(
                RetailResult(
                    source="Best Buy",
                    title=str(product.get("name") or ""),
                    url=str(product.get("url") or ""),
                    price=_float(product.get("salePrice")),
                    identifier=str(product.get("sku") or ""),
                    attributes={
                        "manufacturer": product.get("manufacturer"),
                        "rating": product.get("customerReviewAverage"),
                        "image": product.get("image"),
                    },
                )
            )
        return results


def _float(value: object) -> float | None:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None
