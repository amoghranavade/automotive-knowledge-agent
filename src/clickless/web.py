from pathlib import Path

from datetime import datetime, timezone

from flask import Flask, jsonify, render_template, request

from .llm import DEFAULT_QWEN_MODEL, QwenClient, QwenError
from .data import CatalogError, load_catalog
from .retail import BestBuyClient, EbayClient
from .retail.base import RetailError
from .sources import AutoDevClient, MarketCheckClient, SerpApiClient
from .sources.base import SourceError
from .service import recommend


ROOT = Path(__file__).resolve().parents[2]


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(ROOT / "templates"),
        static_folder=str(ROOT / "static"),
    )

    @app.get("/")
    def index():
        try:
            model_name = QwenClient.from_env().model
        except QwenError:
            model_name = DEFAULT_QWEN_MODEL
        return render_template("index.html", model_name=model_name)

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get("/api/data-status")
    def data_status():
        try:
            catalog = load_catalog()
        except CatalogError as error:
            return jsonify({"ready": False, "error": str(error)}), 503
        live_listings = [
            item for item in catalog["listings"] if item.get("data_mode") == "live"
        ]
        live_products = [
            item for item in catalog["parts"] if item.get("data_mode") == "live"
        ]
        return jsonify(
            {
                "ready": True,
                "snapshot": catalog["snapshot"],
                "counts": {
                    name: len(catalog[name])
                    for name in (
                        "vehicles",
                        "listings",
                        "safety_ratings",
                        "recalls",
                        "reviews",
                        "parts",
                    )
                },
                "connectors": {
                    "marketcheck": MarketCheckClient.from_env() is not None,
                    "auto.dev": AutoDevClient.from_env() is not None,
                    "serpapi": SerpApiClient.from_env() is not None,
                    "ebay": EbayClient.from_env() is not None,
                    "bestbuy": BestBuyClient.from_env() is not None,
                    "nhtsa": True,
                },
                "live_records": {
                    "listings": len(live_listings),
                    "products": len(live_products),
                },
                "live_sources": sorted(
                    {
                        str(item["source_name"])
                        for item in live_listings + live_products
                    }
                ),
            }
        )

    @app.get("/api/model-status")
    def model_status():
        try:
            configured = QwenClient.from_env()
        except QwenError as error:
            return jsonify(
                {
                    "configured": False,
                    "available": False,
                    "model": DEFAULT_QWEN_MODEL,
                    "error": str(error),
                }
            )
        client = QwenClient(
            base_url=configured.base_url,
            model=configured.model,
            api_key=configured.api_key,
            timeout=min(configured.timeout, 2.0),
        )
        status = client.status()
        status["available"] = bool(status["available"]) and status.get(
            "model_available"
        ) is not False
        status["endpoint"] = status.pop("base_url")
        return jsonify(status)

    @app.post("/api/recommend")
    def recommendations():
        body = request.get_json(silent=True) or {}
        query = str(body.get("query", "")).strip()
        if not query:
            return jsonify({"error": "Enter a vehicle request."}), 400
        if len(query) > 1000:
            return jsonify({"error": "Keep the request under 1,000 characters."}), 400

        try:
            return jsonify(
                recommend(
                    query,
                    live_nhtsa=bool(body.get("live_nhtsa", False)),
                    use_model=bool(body.get("use_model", True)),
                    configuration=str(body.get("configuration", "C")),
                )
            )
        except ValueError as error:
            return jsonify({"error": str(error)}), 400

    @app.post("/api/live-check")
    def live_check():
        """ClickLess step: search live marketplaces for one recommended car now."""
        body = request.get_json(silent=True) or {}
        vehicle_id = str(body.get("vehicle_id", ""))
        try:
            catalog = load_catalog()
        except CatalogError as error:
            return jsonify({"error": str(error)}), 503
        vehicle = next((item for item in catalog["vehicles"] if item["id"] == vehicle_id), None)
        if vehicle is None:
            return jsonify({"error": "Unknown vehicle."}), 404
        year, make, model = vehicle["year"], vehicle["make"], vehicle["model"]
        checked_at = datetime.now(timezone.utc).isoformat()
        listings, sources, errors = [], [], []
        for name, client in (
            ("MarketCheck", MarketCheckClient.from_env()),
            ("Auto.dev", AutoDevClient.from_env()),
        ):
            if client is None:
                continue
            sources.append(name)
            try:
                for item in client.search(make=make, model=model, year=year, limit=5):
                    listings.append(
                        {
                            "source": name,
                            "title": f"{item.year} {item.make} {item.model} {item.trim}".strip(),
                            "price": item.price,
                            "mileage": item.mileage,
                            "location": item.location,
                            "url": item.url,
                        }
                    )
            except SourceError as error:
                errors.append(f"{name}: {error}")
        ebay = EbayClient.from_env()
        if ebay is not None:
            sources.append("eBay")
            try:
                for item in ebay.search(f"{year} {make} {model}", limit=15, category_ids="6001"):
                    title = item.title.casefold()
                    if str(year) in title and make.casefold() in title and model.casefold() in title:
                        listings.append(
                            {
                                "source": "eBay",
                                "title": item.title,
                                "price": item.price,
                                "mileage": item.attributes.get("mileage"),
                                "location": "",
                                "url": item.url,
                            }
                        )
            except RetailError as error:
                errors.append(f"eBay: {error}")
        listings.sort(key=lambda item: item["price"] if item["price"] is not None else float("inf"))
        return jsonify(
            {
                "vehicle_id": vehicle_id,
                "checked_at": checked_at,
                "sources": sources,
                "listings": listings[:8],
                "errors": errors,
                "message": None if sources else "No live marketplace keys are configured.",
            }
        )

    return app


app = create_app()


def main() -> None:
    app.run(host="127.0.0.1", port=8000, debug=False)


if __name__ == "__main__":
    main()
