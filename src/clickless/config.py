from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - bare Python fallback
    def load_dotenv(dotenv_path: str | os.PathLike[str] | None = None, **_: object) -> bool:
        """Load KEY=VALUE pairs from a .env file into os.environ without overwriting."""
        path = Path(dotenv_path) if dotenv_path is not None else Path(".env")
        if not path.is_file():
            return False
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            os.environ.setdefault(key, value)
        return True


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
