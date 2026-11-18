from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Constraints:
    max_price: int | None = None
    max_mileage: int | None = None
    min_year: int | None = None
    body_style: str | None = None
    drivetrain: str | None = None
    make: str | None = None
    preferences: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Evidence:
    id: str
    type: str
    source: str
    url: str
    observed_at: str
    text: str
    indirect: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Claim:
    text: str
    evidence_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "evidence_ids": list(self.evidence_ids)}
