from __future__ import annotations

import base64
import os
import re
import time
from typing import Any

import requests

import clickless.config  # noqa: F401

from .base import RetailError, RetailResult


class EbayClient:
    TOKEN_URL = "https://api.ebay.com/identity/v1/oauth2/token"
    SEARCH_URL = "https://api.ebay.com/buy/browse/v1/item_summary/search"

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        marketplace_id: str = "EBAY_US",
        timeout: float = 15.0,
    ) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.marketplace_id = marketplace_id
        self.timeout = timeout
        self._token: str | None = None
        self._expires_at = 0.0

    @classmethod
    def from_env(cls) -> "EbayClient | None":
        client_id = os.getenv("EBAY_CLIENT_ID", "").strip()
        secret = os.getenv("EBAY_CLIENT_SECRET", "").strip()
        if not client_id or not secret:
            return None
        return cls(client_id, secret, os.getenv("EBAY_MARKETPLACE_ID", "EBAY_US"))

    def search(self, query: str, limit: int = 20) -> list[RetailResult]:
        response = requests.get(
            self.SEARCH_URL,
            params={"q": query, "limit": min(max(limit, 1), 200)},
            headers={
                "Authorization": f"Bearer {self._access_token()}",
                "X-EBAY-C-MARKETPLACE-ID": self.marketplace_id,
                "Accept": "application/json",
            },
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            raise RetailError(f"eBay search returned HTTP {response.status_code}")
        payload = response.json()
        results = []
        for item in payload.get("itemSummaries", []):
            price = item.get("price", {}).get("value")
            try:
                parsed_price = float(price)
            except (TypeError, ValueError):
                parsed_price = None
            results.append(
                RetailResult(
                    source="eBay",
                    title=str(item.get("title") or ""),
                    url=str(item.get("itemWebUrl") or ""),
                    price=parsed_price,
                    identifier=str(item.get("itemId") or ""),
                    attributes={
                        "condition": item.get("condition"),
                        "location": item.get("itemLocation"),
                        "image": item.get("image", {}).get("imageUrl"),
                        "mileage": _mileage_from_item(item),
                    },
                )
            )
        return results

    def _access_token(self) -> str:
        if self._token and time.time() < self._expires_at - 60:
            return self._token
        basic = base64.b64encode(
            f"{self.client_id}:{self.client_secret}".encode("utf-8")
        ).decode("ascii")
        response = requests.post(
            self.TOKEN_URL,
            headers={
                "Authorization": f"Basic {basic}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={
                "grant_type": "client_credentials",
                "scope": "https://api.ebay.com/oauth/api_scope",
            },
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            raise RetailError(f"eBay authentication returned HTTP {response.status_code}")
        payload: dict[str, Any] = response.json()
        token = payload.get("access_token")
        if not token:
            raise RetailError("eBay authentication did not return an access token")
        self._token = str(token)
        self._expires_at = time.time() + int(payload.get("expires_in") or 7200)
        return self._token


def _mileage(title: str) -> int | None:
    match = re.search(r"\b([\d,]{2,7})\s*(?:miles|mi)\b", title, re.I)
    return int(match.group(1).replace(",", "")) if match else None


def _mileage_from_item(item: dict[str, Any]) -> int | None:
    for aspect in item.get("localizedAspects", []):
        if str(aspect.get("name") or "").casefold() in {"mileage", "odometer"}:
            value = _mileage(f"{aspect.get('value')} miles")
            if value is not None:
                return value
    return _mileage(str(item.get("title") or ""))
