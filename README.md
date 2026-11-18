# ClickLess Automotive

ClickLess Automotive turns a natural language vehicle request into ranked used-car options with listing, owner-review, aftermarket-product, NHTSA safety/recall, and GraphRAG citation evidence.

The current research snapshot contains:

- 30 make-model-year-drivetrain variants
- 969 Craigslist listing records from the Kaggle Used Cars Dataset plus a validated Auto.dev live record
- 2,014 OpinRank and Amazon Automotive reviews
- 300 Amazon Automotive product records plus three validated SerpApi live offers
- 23 matched NHTSA crash-rating records
- 196 NHTSA recall records

The application reads its catalog from `data/processed/catalog.json`.

## Setup

```bash
git clone https://github.com/amoghranavade/automotive-knowledge-agent.git
cd automotive-knowledge-agent
python3 -m venv .venv
source .venv/bin/activate  # On Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e '.[data,graph,ml]'
```

Optional extras:

- **Browser capture (Playwright, Selenium, Scrapy)**:
  ```bash
  pip install -e '.[crawl]'
  playwright install chromium
  ```
- **Test runner**:
  ```bash
  pip install -e '.[dev]'
  ```
- **Everything including crawl/graph/ml**:
  ```bash
  pip install -e '.[all]' pytest
  ```

Cache MiniLM once before running configuration B–E:

```bash
python scripts/prepare_models.py
```

## Environment file

Copy `.env.example` to `.env` and add the credentials you use.

```dotenv
CLICKLESS_QWEN_MODEL=qwen38-27b
CLICKLESS_QWEN_API_KEY=your_actual_key_here
```

Optional graph and retailer settings:

```dotenv
CLICKLESS_GRAPH_BACKEND=memory
CLICKLESS_RETRIEVAL_MODE=graph
MARKETCHECK_API_KEY=
AUTO_DEV_API_KEY=
SERPAPI_API_KEY=
EBAY_CLIENT_ID=
EBAY_CLIENT_SECRET=
BESTBUY_API_KEY=
```

Now only `AUTO_DEV_API_KEY` and `SERPAPI_API_KEY` are needed for the current live data path. MarketCheck, eBay, and Best Buy are optional; connectors without credentials are skipped.

The `.env` file is ignored by Git. Never commit or send it.

## Build the data snapshot

The builder downloads the public sources, keeps a controlled subset, cleans the records, adds provenance, and retrieves NHTSA data:

```bash
clickless-build-data --download-public
```

Output:

```text
data/processed/catalog.json
```

The Kaggle dataset downloads through the official CLI when available and falls back to direct HTTP download if needed.

## Run tests

Run the unit and integration test suite with `pytest`:

```bash
pytest
```

All tests mock paid external search APIs (`Auto.dev`, `MarketCheck`, `SerpApi`, `eBay`, `Best Buy`) and run locally against deterministic fixtures.

## Run the web application (Flask)

```bash
clickless
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

- In Chrome or Edge, select **Just say it** to dictate the vehicle request. Typed input always remains available.
- When budget or body style is unspecified, a follow-up prompt appears with an **Add my reply to the last search** toggle so you can refine the request in one step.
- Open **Settings** to toggle the AI helper, toggle live NHTSA recall lookups, or switch between evaluation configurations `A`–`E`.
- Each candidate card displays an overall match gauge, color-coded 5-segment ratings (`Safety`, `Reliability`, `Upkeep cost`, `Fuel economy`, `Cargo space`, `Winter traction`) with `No data` states, crash/recall/live badges, owner review highlights, a **Check live listings** button, and an expandable **How we picked this** drawer showing the pipeline method, owner aspect bundles, recalls, GraphRAG paths, and retrieved passages.

### Web API endpoints

- `GET /api/health`: returns `{"status": "ok"}`.
- `GET /api/data-status`: returns catalog snapshot counts, connector availability, `graph_backend`, and `retrieval_mode` (returns HTTP `503` if the catalog cannot be loaded).
- `GET /api/model-status`: checks Qwen endpoint availability and configured model.
- `POST /api/recommend`: accepts JSON `{"query": "...", "configuration": "C", "use_model": true, "live_nhtsa": false, "candidate_limit": 3}` (`candidate_limit` optional `1..10`; returns `400` on invalid input, `503` on missing/invalid catalog, `500` on verification failure).
- `POST /api/live-check`: accepts JSON `{"vehicle_id": "..."}` and queries MarketCheck, Auto.dev, and eBay (whichever keys are set) for that exact year, make, and model with a UTC timestamp (`404` for unknown `vehicle_id`, `503` if catalog cannot be loaded).

## System configurations

The web UI and the evaluation runner expose the five proposal configurations:

- **A**: rule parsing, BM25 retrieval, and domain-tuned 4-class VADER aspect sentiment
- **B**: flat `all-MiniLM-L6-v2` retrieval without graph paths
- **C**: full ontology GraphRAG retrieval with Qwen aspect extraction and grounded summaries when connected
- **D**: GraphRAG retrieval over raw reviews without structured aspect-assertion nodes
- **E**: GraphRAG retrieval with sentiment aspect weights removed from ranking

Hard price, mileage, year, make, body-style, and drivetrain filters remain deterministic in every configuration.

## Live data refresh and crawlers

After adding MarketCheck, Auto.dev, or SerpApi credentials to `.env`:

```bash
python scripts/refresh_retail.py --per-vehicle 1 --max-vehicles 3
```

MarketCheck is the primary live vehicle source when configured and Auto.dev is its fallback. SerpApi supplies live aftermarket offers across Google Shopping merchants. Live records are timestamped and labeled `Live`; research records are labeled `Frozen snapshot`. Direct eBay and Best Buy refreshes run only when their credentials are present.

One-page retail capture with JSON-LD (`jsonld-v1`) and CSS DOM fallback (`dom-css-v2`) extraction supports Playwright, Selenium, and Scrapy engines:

```bash
clickless-crawl ebay "2009 Subaru Forester" --engine playwright
```

Import pages captured with `clickless-crawl` into the catalog (they appear as `Live capture` records):

```bash
clickless-crawl walmart "2009 Toyota RAV4 battery"
python scripts/import_captures.py
```

Refresh NHTSA crash-test ratings and recalls (including pre-2011 per-test averages and model/make aliases) without rebuilding the entire catalog:

```bash
python scripts/refresh_safety.py --report
python scripts/refresh_safety.py
```

During the demo, each recommended car has a **Check live listings** button. It queries MarketCheck, Auto.dev, and eBay (whichever keys are set) for that exact year, make, and model and shows the results with a timestamp.

## Neo4j and RDF

Start Neo4j and load the catalog:

```bash
docker compose up -d neo4j
python scripts/load_neo4j.py --clear
```

Then set `CLICKLESS_GRAPH_BACKEND=neo4j` in `.env` and restart the app. If Neo4j is unavailable, retrieval falls back to the same ontology graph in memory and labels the returned paths as `memory_fallback`.

Export and validate the Schema.org/PROV-O RDF graph against `data/shapes.ttl`:

```bash
python scripts/export_rdf.py
```

The Turtle export is written to `data/processed/clickless.ttl`.

## Evaluation

Run the 50-scenario (15 development / 35 test) system benchmark across configurations A–E:

```bash
clickless-evaluate --configurations ABCDE
```

Add `--use-model` after `CLICKLESS_QWEN_API_KEY` is configured to evaluate Qwen parsing, aspect extraction, and grounded summaries.

The evaluation writes to `evaluation/output/`:

- `system_results.jsonl`
- `summary.json` (overall and per-configuration constraint satisfaction, gold constraint satisfaction, abstention accuracy, cohort retention, citation coverage, median & p95 latency)
- `human_judgments.csv`, `citation_audit.csv`, `explanation_ratings.csv`, `blinding_key.json`
- `absa_annotations.csv`

After judges fill `human_judgments.csv`, calculate Precision@5, nDCG@5, Cohen's kappa, paired bootstrap confidence intervals, and ABSA span/polarity metrics:

```bash
clickless-analyze-evaluation
```

Add `--use-model` to `clickless-analyze-evaluation` to score Qwen ABSA against the annotated test sentences.

Run component audits (50-query slot-parser benchmark, 100-link graph audit sheet, SHACL validation, and web-extraction audit):

```bash
clickless-audit-components
```

Pass `--use-model` to benchmark Qwen slot parsing alongside the deterministic rule parser, or `--overwrite` to regenerate existing audit sheets.
