from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


def _path(name: str, default: str) -> Path:
    value = Path(os.getenv(name, default))
    return value if value.is_absolute() else ROOT / value


def _boolean(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    root: Path = ROOT
    catalog_path: Path = _path("CLICKLESS_CATALOG_PATH", "data/processed/catalog.json")
    downloads_path: Path = ROOT / "data" / "downloads"
    processed_path: Path = ROOT / "data" / "processed"
    live_nhtsa: bool = _boolean("CLICKLESS_LIVE_NHTSA", True)
    retrieval_mode: str = os.getenv("CLICKLESS_RETRIEVAL_MODE", "graph")


settings = Settings()
