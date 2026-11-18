from __future__ import annotations

import os

from .base import RetailError


class WalmartClient:
    def __init__(self, client_id: str, client_secret: str) -> None:
        self.client_id = client_id
        self.client_secret = client_secret

    @classmethod
    def from_env(cls) -> "WalmartClient | None":
        client_id = os.getenv("WALMART_CLIENT_ID", "").strip()
        secret = os.getenv("WALMART_CLIENT_SECRET", "").strip()
        return cls(client_id, secret) if client_id and secret else None

    def search(self, query: str, limit: int = 10) -> list[object]:
        raise RetailError(
            "Walmart Marketplace credentials do not provide a general buyer catalog search; "
            "use an approved frozen snapshot."
        )
