from __future__ import annotations

from html.parser import HTMLParser
import json
import re
from typing import Any, Iterable
from urllib.parse import urljoin

try:
    from parsel import Selector as _ParselSelector
except ImportError:  # pragma: no cover - optional crawl dependency
    _ParselSelector = None


SITE_ORIGINS = {
    "ebay": "https://www.ebay.com",
    "amazon": "https://www.amazon.com",
    "walmart": "https://www.walmart.com",
    "bestbuy": "https://www.bestbuy.com",
}

SEARCH_URLS = {
    "ebay": "https://www.ebay.com/sch/i.html?_nkw={query}",
    "amazon": "https://www.amazon.com/s?k={query}",
    "walmart": "https://www.walmart.com/search?q={query}",
    "bestbuy": "https://www.bestbuy.com/site/searchpage.jsp?st={query}",
}

DOM_SPECS: dict[str, dict[str, Any]] = {
    "ebay": {
        "container": ".s-item",
        "name": (".s-item__title", "h3", '[role="heading"]', "a.s-item__link", "a"),
        "url": ("a.s-item__link", "a"),
        "price": (".s-item__price", '[data-testid="price"]', ".price"),
        "rating": (".x-star-rating", '[aria-label*="out of 5"]', ".rating"),
        "skip_names": {"shop on ebay"},
    },
    "amazon": {
        "container": '[data-component-type="s-search-result"]',
        "name": ("h2 a span", "h2 span", "h2", ".a-text-normal", "a.a-link-normal"),
        "url": ("h2 a", "a.a-link-normal", "a"),
        "price": (".a-price .a-offscreen", ".a-price", ".a-price-whole", ".price"),
        "rating": (
            "i.a-icon-star-small span.a-icon-alt",
            ".a-icon-alt",
            '[aria-label*="out of 5"]',
            ".rating",
        ),
        "skip_names": set(),
    },
    "walmart": {
        "container": "[data-item-id]",
        "name": (
            '[data-automation-id="product-title"]',
            "span.lh-title",
            "a span",
            "h2",
            "h3",
            "a",
        ),
        "url": ("a[link-identifier]", "a"),
        "price": (
            '[data-automation-id="product-price"]',
            '[itemprop="price"]',
            '[aria-label*="$"]',
            ".price",
        ),
        "rating": (
            '[data-testid="product-ratings"]',
            '[aria-label*="out of 5"]',
            ".stars-reviews-count-node",
            ".rating",
        ),
        "skip_names": set(),
    },
    "bestbuy": {
        "container": ".sku-item",
        "name": (".sku-title a", ".sku-header a", "h4.sku-title", "h4 a", "h4", "h3", "a"),
        "url": (".sku-title a", ".sku-header a", "h4 a", "a"),
        "price": (
            '.priceView-hero-price span[aria-hidden="true"]',
            ".priceView-hero-price span",
            ".priceView-customer-price span",
            '[data-testid="customer-price"]',
            ".price",
        ),
        "rating": (
            ".c-ratings-reviews .ugc-c-review-average",
            "p.visually-hidden",
            '[aria-label*="out of 5"]',
            ".rating",
        ),
        "skip_names": set(),
    },
}


def extract_products(html: str, site: str, captured_at: str) -> list[dict[str, Any]]:
    site_key = site.strip().lower()
    if _ParselSelector is not None:
        products = _extract_jsonld_parsel(html, site_key, captured_at)
        if not products:
            products = _extract_dom_parsel(html, site_key, captured_at)
    else:
        root = _parse_html_tree(html)
        products = _extract_jsonld_tree(root, site_key, captured_at)
        if not products:
            products = _extract_dom_tree(root, site_key, captured_at)

    unique: dict[str, dict[str, Any]] = {}
    for product in products:
        key = product.get("url") or f"{product.get('name')}:{product.get('price')}"
        unique[str(key)] = product
    return list(unique.values())


def _extract_jsonld_parsel(
    html: str,
    site: str,
    captured_at: str,
) -> list[dict[str, Any]]:
    selector = _ParselSelector(text=html)
    blocks = selector.css('script[type="application/ld+json"]::text').getall()
    return _jsonld_blocks_to_products(blocks, site, captured_at)


def _extract_jsonld_tree(
    root: "_HtmlNode",
    site: str,
    captured_at: str,
) -> list[dict[str, Any]]:
    blocks = [
        node.raw_text()
        for node in root.select('script[type="application/ld+json"]')
    ]
    return _jsonld_blocks_to_products(blocks, site, captured_at)


def _jsonld_blocks_to_products(
    blocks: Iterable[str],
    site: str,
    captured_at: str,
) -> list[dict[str, Any]]:
    products: list[dict[str, Any]] = []
    for raw in blocks:
        raw_text = raw.strip()
        if not raw_text:
            continue
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            continue
        for item in _walk(payload):
            raw_type = item.get("@type")
            types = raw_type if isinstance(raw_type, list) else [raw_type]
            if not any(t in {"Product", "Vehicle"} for t in types):
                continue
            name = _clean_text(item.get("name"))
            if not name:
                continue
            offer = item.get("offers") or {}
            if isinstance(offer, list):
                offer = offer[0] if offer else {}
            if not isinstance(offer, dict):
                offer = {}
            products.append(
                {
                    "name": name,
                    "url": _resolve_url(site, item.get("url") or offer.get("url")),
                    "price": _parse_price(offer.get("price") or offer.get("lowPrice")),
                    "currency": offer.get("priceCurrency"),
                    "rating": _parse_rating(
                        (item.get("aggregateRating") or {}).get("ratingValue")
                        if isinstance(item.get("aggregateRating"), dict)
                        else None
                    ),
                    "source": site,
                    "captured_at": captured_at,
                    "selector_version": "jsonld-v1",
                    "data_mode": "live_capture",
                }
            )
    return products


def _extract_dom_parsel(
    html: str,
    site: str,
    captured_at: str,
) -> list[dict[str, Any]]:
    spec = DOM_SPECS.get(site)
    if spec is None:
        return []
    selector = _ParselSelector(text=html)
    products: list[dict[str, Any]] = []
    for card in selector.css(spec["container"]):
        name = _first_parsel_text(card, spec["name"])
        if not name or name.casefold() in spec["skip_names"]:
            continue
        raw_url = _first_parsel_attr(card, spec["url"], "href")
        url = _resolve_url(site, raw_url)
        price = _first_parsel_price(card, spec["price"])
        rating = _first_parsel_rating(card, spec["rating"])
        products.append(
            {
                "name": name,
                "url": url,
                "price": price,
                "currency": "USD" if price is not None else None,
                "rating": rating,
                "source": site,
                "captured_at": captured_at,
                "selector_version": "dom-css-v2",
                "data_mode": "live_capture",
            }
        )
    return products


def _first_parsel_text(card: Any, css_list: tuple[str, ...]) -> str | None:
    for css in css_list:
        for node in card.css(css):
            text = _clean_text(" ".join(node.css("::text").getall()))
            if text:
                return text
    return None


def _first_parsel_attr(card: Any, css_list: tuple[str, ...], attr: str) -> str | None:
    for css in css_list:
        for value in card.css(f"{css}::attr({attr})").getall():
            cleaned = _clean_text(value)
            if cleaned and cleaned != "#" and not cleaned.lower().startswith("javascript:"):
                return cleaned
    return None


def _first_parsel_price(card: Any, css_list: tuple[str, ...]) -> float | None:
    for css in css_list:
        for node in card.css(css):
            compact = _clean_text("".join(node.css("::text").getall()))
            parsed = _parse_price(compact)
            if parsed is not None:
                return parsed
            for attr in ("content", "aria-label"):
                attr_val = node.attrib.get(attr)
                parsed = _parse_price(attr_val)
                if parsed is not None:
                    return parsed
    return None


def _first_parsel_rating(card: Any, css_list: tuple[str, ...]) -> float | None:
    for css in css_list:
        for node in card.css(css):
            text = _clean_text(" ".join(node.css("::text").getall()))
            parsed = _parse_rating(text)
            if parsed is not None:
                return parsed
            for attr in ("aria-label", "content", "title"):
                parsed = _parse_rating(node.attrib.get(attr))
                if parsed is not None:
                    return parsed
    return None


def _extract_dom_tree(
    root: "_HtmlNode",
    site: str,
    captured_at: str,
) -> list[dict[str, Any]]:
    spec = DOM_SPECS.get(site)
    if spec is None:
        return []
    products: list[dict[str, Any]] = []
    for card in root.select(spec["container"]):
        name = _first_tree_text(card, spec["name"])
        if not name or name.casefold() in spec["skip_names"]:
            continue
        raw_url = _first_tree_attr(card, spec["url"], "href")
        url = _resolve_url(site, raw_url)
        price = _first_tree_price(card, spec["price"])
        rating = _first_tree_rating(card, spec["rating"])
        products.append(
            {
                "name": name,
                "url": url,
                "price": price,
                "currency": "USD" if price is not None else None,
                "rating": rating,
                "source": site,
                "captured_at": captured_at,
                "selector_version": "dom-css-v2",
                "data_mode": "live_capture",
            }
        )
    return products


def _first_tree_text(card: "_HtmlNode", css_list: tuple[str, ...]) -> str | None:
    for css in css_list:
        for node in card.select(css):
            text = _clean_text(node.text())
            if text:
                return text
    return None


def _first_tree_attr(card: "_HtmlNode", css_list: tuple[str, ...], attr: str) -> str | None:
    for css in css_list:
        for node in card.select(css):
            cleaned = _clean_text(node.attrs.get(attr))
            if cleaned and cleaned != "#" and not cleaned.lower().startswith("javascript:"):
                return cleaned
    return None


def _first_tree_price(card: "_HtmlNode", css_list: tuple[str, ...]) -> float | None:
    for css in css_list:
        for node in card.select(css):
            parsed = _parse_price(_clean_text(node.compact_text()))
            if parsed is not None:
                return parsed
            for attr in ("content", "aria-label"):
                parsed = _parse_price(node.attrs.get(attr))
                if parsed is not None:
                    return parsed
    return None


def _first_tree_rating(card: "_HtmlNode", css_list: tuple[str, ...]) -> float | None:
    for css in css_list:
        for node in card.select(css):
            parsed = _parse_rating(_clean_text(node.text()))
            if parsed is not None:
                return parsed
            for attr in ("aria-label", "content", "title"):
                parsed = _parse_rating(node.attrs.get(attr))
                if parsed is not None:
                    return parsed
    return None


def _resolve_url(site: str, raw_url: Any) -> str | None:
    text = _clean_text(raw_url)
    if not text or text == "#" or text.lower().startswith("javascript:"):
        return None
    origin = SITE_ORIGINS.get(site, "")
    if origin and not text.startswith(("http://", "https://")):
        return urljoin(origin, text)
    return text


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = re.sub(r"\s+", " ", str(value)).strip()
    return cleaned or None


def _parse_price(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    match = re.search(r"(\d[\d,]*(?:\.\d+)?)", str(value))
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return None


def _parse_rating(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        num = float(value)
        return num if 0.0 <= num <= 5.0 else None
    match = re.search(r"(\d+(?:\.\d+)?)", str(value))
    if not match:
        return None
    try:
        num = float(match.group(1))
    except ValueError:
        return None
    return num if 0.0 <= num <= 5.0 else None


def _walk(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


# ---------------------------------------------------------------------------
# Pure-Python HTML DOM + CSS selector fallback when parsel is not installed
# ---------------------------------------------------------------------------

VOID_ELEMENTS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
}
_SIMPLE_SELECTOR_TOKEN = re.compile(
    r'([.#]?[A-Za-z0-9_-]+)|(\[[A-Za-z0-9_:-]+(?:[*^$]?=(?:"[^"]*"|\'[^\']*\'|[^\]]+))?\])'
)


class _HtmlNode:
    def __init__(self, tag: str, attrs: dict[str, str] | None = None) -> None:
        self.tag = tag.lower()
        self.attrs: dict[str, str] = attrs or {}
        self.children: list[_HtmlNode] = []
        self.text_parts: list[str] = []
        self.classes: set[str] = {
            part for part in self.attrs.get("class", "").split() if part
        }

    def raw_text(self) -> str:
        chunks = list(self.text_parts)
        for child in self.children:
            chunks.append(child.raw_text())
        return "".join(chunks)

    def text(self) -> str:
        chunks = list(self.text_parts)
        for child in self.children:
            chunks.append(child.text())
        return " ".join(chunk for chunk in chunks if chunk)

    def compact_text(self) -> str:
        chunks = list(self.text_parts)
        for child in self.children:
            chunks.append(child.compact_text())
        return "".join(chunks)

    def iter_descendants(self) -> Iterable["_HtmlNode"]:
        for child in self.children:
            yield child
            yield from child.iter_descendants()

    def select(self, selector: str) -> list["_HtmlNode"]:
        parts = [part.strip() for part in selector.split() if part.strip()]
        if not parts:
            return []
        current: list[_HtmlNode] = [self]
        for compound in parts:
            matched: list[_HtmlNode] = []
            seen: set[int] = set()
            for ancestor in current:
                for node in ancestor.iter_descendants():
                    if id(node) not in seen and _matches_compound(node, compound):
                        seen.add(id(node))
                        matched.append(node)
            current = matched
        return current


def _matches_compound(node: _HtmlNode, compound: str) -> bool:
    pos = 0
    for match in _SIMPLE_SELECTOR_TOKEN.finditer(compound):
        if match.start() != pos:
            return False
        pos = match.end()
        token = match.group(0)
        if token.startswith("."):
            if token[1:] not in node.classes:
                return False
        elif token.startswith("#"):
            if node.attrs.get("id") != token[1:]:
                return False
        elif token.startswith("[") and token.endswith("]"):
            inner = token[1:-1]
            if "*=" in inner:
                attr, expected = inner.split("*=", 1)
                expected = expected.strip().strip("\"'")
                if expected not in node.attrs.get(attr.strip().lower(), ""):
                    return False
            elif "=" in inner:
                attr, expected = inner.split("=", 1)
                expected = expected.strip().strip("\"'")
                if node.attrs.get(attr.strip().lower()) != expected:
                    return False
            else:
                if inner.strip().lower() not in node.attrs:
                    return False
        else:
            if node.tag != token.lower():
                return False
    return pos == len(compound)


class _DOMBuilder(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = _HtmlNode("document")
        self.stack: list[_HtmlNode] = [self.root]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_l = tag.lower()
        attr_map = {k.lower(): (v if v is not None else "") for k, v in attrs}
        node = _HtmlNode(tag_l, attr_map)
        self.stack[-1].children.append(node)
        if tag_l not in VOID_ELEMENTS:
            self.stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_l = tag.lower()
        attr_map = {k.lower(): (v if v is not None else "") for k, v in attrs}
        self.stack[-1].children.append(_HtmlNode(tag_l, attr_map))

    def handle_endtag(self, tag: str) -> None:
        tag_l = tag.lower()
        for idx in range(len(self.stack) - 1, 0, -1):
            if self.stack[idx].tag == tag_l:
                del self.stack[idx:]
                break

    def handle_data(self, data: str) -> None:
        if data:
            self.stack[-1].text_parts.append(data)


def _parse_html_tree(html: str) -> _HtmlNode:
    parser = _DOMBuilder()
    parser.feed(html)
    parser.close()
    return parser.root
