from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class RetailError(RuntimeError):
    pass


@dataclass(frozen=True)
class RetailResult:
    source: str
    title: str
    url: str
    price: float | None
    identifier: str
    attributes: dict[str, Any]
    data_mode: str = "live"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "title": self.title,
            "url": self.url,
            "price": self.price,
            "identifier": self.identifier,
            "attributes": self.attributes,
            "data_mode": self.data_mode,
        }
