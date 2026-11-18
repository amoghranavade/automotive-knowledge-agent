from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote_plus

from clickless.config import settings
from .extract import extract_products


SEARCH_URLS = {
    "ebay": "https://www.ebay.com/sch/i.html?_nkw={query}",
    "amazon": "https://www.amazon.com/s?k={query}",
    "walmart": "https://www.walmart.com/search?q={query}",
    "bestbuy": "https://www.bestbuy.com/site/searchpage.jsp?st={query}",
}


async def capture(site: str, query: str, output: Path) -> dict:
    try:
        from playwright.async_api import async_playwright
    except ImportError as error:
        raise RuntimeError("Install crawl dependencies: pip install -e '.[crawl]'") from error
    url = SEARCH_URLS[site].format(query=quote_plus(query))
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        page = await browser.new_page(locale="en-US")
        await page.goto(url, wait_until="domcontentloaded", timeout=45_000)
        await page.wait_for_timeout(2_000)
        html = await page.content()
        await browser.close()

    timestamp = datetime.now(timezone.utc).isoformat()
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{site}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    html_path = output / f"{stem}.html"
    json_path = output / f"{stem}.json"
    html_path.write_text(html, encoding="utf-8")
    payload = {
        "site": site,
        "query": query,
        "url": url,
        "captured_at": timestamp,
        "selector_version": "jsonld-v1",
        "products": extract_products(html, site, timestamp),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"html": str(html_path), "records": str(json_path), "count": len(payload["products"])}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture one permitted retail search page and extract JSON-LD products"
    )
    parser.add_argument("site", choices=sorted(SEARCH_URLS))
    parser.add_argument("query")
    parser.add_argument("--output", default=settings.root / "data" / "cache" / "retail")
    args = parser.parse_args()
    print(json.dumps(asyncio.run(capture(args.site, args.query, Path(args.output))), indent=2))


if __name__ == "__main__":
    main()
