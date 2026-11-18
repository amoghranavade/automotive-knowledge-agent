from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

from clickless.config import settings
from .extract import SEARCH_URLS, extract_products
from .spiders import run_scrapy_capture


async def _fetch_playwright_html(url: str) -> str:
    try:
        from playwright.async_api import async_playwright
    except ImportError as error:
        raise RuntimeError("Install crawl dependencies: pip install -e '.[crawl]'") from error
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page(locale="en-US")
            await page.goto(url, wait_until="domcontentloaded", timeout=45_000)
            await page.wait_for_timeout(2_000)
            return await page.content()
        finally:
            await browser.close()


def _fetch_selenium_html(url: str) -> str:
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
    except ImportError as error:
        raise RuntimeError("Install crawl dependencies: pip install -e '.[crawl]'") from error
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-gpu")
    options.add_argument("--lang=en-US")
    driver = webdriver.Chrome(options=options)
    try:
        driver.set_page_load_timeout(45)
        driver.get(url)
        return driver.page_source
    finally:
        driver.quit()


async def capture(
    site: str,
    query: str,
    output: Path,
    engine: str = "playwright",
) -> dict[str, Any]:
    site_key = site.strip().lower()
    if site_key not in SEARCH_URLS:
        raise ValueError(f"Unsupported site: {site}")
    engine_key = engine.strip().lower()
    if engine_key == "scrapy":
        return await asyncio.to_thread(run_scrapy_capture, site_key, query, output)

    url = SEARCH_URLS[site_key].format(query=quote_plus(query))
    if engine_key == "playwright":
        html = await _fetch_playwright_html(url)
    elif engine_key == "selenium":
        html = await asyncio.to_thread(_fetch_selenium_html, url)
    else:
        raise ValueError(f"Unsupported capture engine: {engine}")

    timestamp = datetime.now(timezone.utc).isoformat()
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{site_key}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    html_path = output / f"{stem}.html"
    json_path = output / f"{stem}.json"
    html_path.write_text(html, encoding="utf-8")
    products = extract_products(html, site_key, timestamp)
    selector_version = (
        products[0]["selector_version"] if products else "jsonld-v1 / dom-css-v2"
    )
    payload = {
        "site": site_key,
        "query": query,
        "url": url,
        "captured_at": timestamp,
        "engine": engine_key,
        "selector_version": selector_version,
        "products": products,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "html": str(html_path),
        "records": str(json_path),
        "count": len(products),
        "engine": engine_key,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture one permitted retail search page and extract products"
    )
    parser.add_argument("site", choices=sorted(SEARCH_URLS))
    parser.add_argument("query")
    parser.add_argument(
        "--engine",
        choices=("playwright", "selenium", "scrapy"),
        default="playwright",
    )
    parser.add_argument("--output", default=settings.root / "data" / "cache" / "retail")
    args = parser.parse_args()
    print(
        json.dumps(
            asyncio.run(capture(args.site, args.query, Path(args.output), engine=args.engine)),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
