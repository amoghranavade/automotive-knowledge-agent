from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class SourceError(RuntimeError):
    pass


@dataclass(frozen=True)
class VehicleListing:
    source: str
    identifier: str
    year: int
    make: str
    model: str
    trim: str
    price: int
    mileage: int
    location: str
    vin: str
    url: str
    captured_at: str
    attributes: dict[str, Any]


@dataclass(frozen=True)
class ProductOffer:
    source: str
    identifier: str
    title: str
    price: float
    url: str
    merchant: str
    rating: float | None
    review_count: int | None
    review_texts: tuple[str, ...]
    captured_at: str
    attributes: dict[str, Any]
