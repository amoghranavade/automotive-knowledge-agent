from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from typing import Any, Iterable
from urllib.parse import quote_plus

from .extract import SEARCH_URLS, extract_products

try:
    import scrapy
    from scrapy.crawler import CrawlerProcess

    SCRAPY_AVAILABLE = True
except ImportError:  # pragma: no cover - optional crawl dependency
    scrapy = None
    CrawlerProcess = None
    SCRAPY_AVAILABLE = False


_SpiderBase = scrapy.Spider if SCRAPY_AVAILABLE else object


class RetailListingSpider(_SpiderBase):
    """Scrapy spider that extracts provenance-tagged retail products from search HTML."""

    name = "retail_listings"
    custom_settings = {
        "ROBOTSTXT_OBEY": True,
        "LOG_ENABLED": False,
        "USER_AGENT": "ClickLessAutomotive/1.0",
    }

    def __init__(
        self,
        site: str = "ebay",
        query: str = "",
        url: str | None = None,
        captured_at: str | None = None,
        collected: list[dict[str, Any]] | None = None,
        html_snapshots: list[str] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if SCRAPY_AVAILABLE:
            super().__init__(*args, **kwargs)
        site_key = site.strip().lower()
        if site_key not in SEARCH_URLS:
            raise ValueError(f"Unsupported site for RetailListingSpider: {site}")
        self.site = site_key
        self.query = query
        self.captured_at = captured_at
        self.collected = collected if collected is not None else []
        self.html_snapshots = html_snapshots if html_snapshots is not None else []
        target_url = url or SEARCH_URLS[site_key].format(query=quote_plus(query))
        self.start_urls = [target_url]

    def parse(self, response: Any) -> Iterable[dict[str, Any]]:
        timestamp = self.captured_at or datetime.now(timezone.utc).isoformat()
        html = getattr(response, "text", None)
        if html is None:
            body = getattr(response, "body", b"")
            html = body.decode("utf-8", errors="replace") if isinstance(body, (bytes, bytearray)) else str(body)
        self.html_snapshots.append(html)
        page_url = getattr(response, "url", None) or (self.start_urls[0] if self.start_urls else "")
        products = extract_products(html, self.site, timestamp)
        for product in products:
            item = {
                **product,
                "query": self.query,
                "page_url": page_url,
                "spider": self.name,
                "engine": "scrapy",
            }
            self.collected.append(item)
            yield item


def run_scrapy_capture(
    site: str,
    query: str,
    output: Path,
    url: str | None = None,
) -> dict[str, Any]:
    if not SCRAPY_AVAILABLE or CrawlerProcess is None:
        raise RuntimeError("Install crawl dependencies: pip install -e '.[crawl]'")
    site_key = site.strip().lower()
    if site_key not in SEARCH_URLS:
        raise ValueError(f"Unsupported site for Scrapy capture: {site}")

    target_url = url or SEARCH_URLS[site_key].format(query=quote_plus(query))
    timestamp = datetime.now(timezone.utc).isoformat()
    collected: list[dict[str, Any]] = []
    html_snapshots: list[str] = []

    process = CrawlerProcess(settings={"LOG_ENABLED": False})
    process.crawl(
        RetailListingSpider,
        site=site_key,
        query=query,
        url=target_url,
        captured_at=timestamp,
        collected=collected,
        html_snapshots=html_snapshots,
    )
    process.start(
        install_signal_handlers=threading.current_thread() is threading.main_thread()
    )

    output.mkdir(parents=True, exist_ok=True)
    stem = f"{site_key}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    html_path = output / f"{stem}.html"
    json_path = output / f"{stem}.json"
    html_text = html_snapshots[0] if html_snapshots else ""
    html_path.write_text(html_text, encoding="utf-8")

    selector_version = (
        collected[0]["selector_version"] if collected else "jsonld-v1 / dom-css-v2"
    )
    payload = {
        "site": site_key,
        "query": query,
        "url": target_url,
        "captured_at": timestamp,
        "engine": "scrapy",
        "selector_version": selector_version,
        "products": collected,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "html": str(html_path),
        "records": str(json_path),
        "count": len(collected),
        "engine": "scrapy",
    }
