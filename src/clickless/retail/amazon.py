from __future__ import annotations

import os

from .base import RetailError


class AmazonCreatorsClient:
    def __init__(self, client_id: str, client_secret: str, partner_tag: str) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.partner_tag = partner_tag

    @classmethod
    def from_env(cls) -> "AmazonCreatorsClient | None":
        values = (
            os.getenv("AMAZON_CREATORS_CLIENT_ID", "").strip(),
            os.getenv("AMAZON_CREATORS_CLIENT_SECRET", "").strip(),
            os.getenv("AMAZON_PARTNER_TAG", "").strip(),
        )
        return cls(*values) if all(values) else None

    def search(self, query: str, limit: int = 10) -> list[object]:
        raise RetailError(
            "Amazon Creators API access is not configured in this coursework build; "
            "use the licensed Amazon Reviews 2023 snapshot."
        )
