# ClickLess Automotive

ClickLess Automotive turns a vehicle request into three ranked used-car options with listing, owner-review, product, safety, recall, and source evidence.

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
source .venv/bin/activate
pip install -e '.[data,graph,ml]'
```

For browser capture support:

```bash
pip install -e '.[crawl]'
playwright install chromium
```

Cache MiniLM once before running configuration B-E:

```bash
python scripts/prepare_models.py
```

## Environment file

Copy `.env.example` to `.env` and add the credentials you use.

Paste the ASU key after the equals sign:

```dotenv
CLICKLESS_QWEN_MODEL=qwen38-27b
CLICKLESS_QWEN_API_KEY=your_actual_key_here
```

Optional retailer keys use these lines:

```dotenv
MARKETCHECK_API_KEY=
AUTO_DEV_API_KEY=
SERPAPI_API_KEY=
EBAY_CLIENT_ID=
EBAY_CLIENT_SECRET=
BESTBUY_API_KEY=
```

Only `AUTO_DEV_API_KEY` and `SERPAPI_API_KEY` are needed for the current live data path. MarketCheck, eBay, and Best Buy are optional; connectors without credentials are skipped.

The `.env` file is ignored by Git. Never commit or send it.

## Build the data snapshot

The builder downloads the public sources, keeps a controlled subset, cleans the records, adds provenance, and retrieves NHTSA data:

```bash
source .venv/bin/activate
clickless-build-data --download-public
```

Output:

```text
data/processed/catalog.json
```

The Kaggle dataset is public and currently downloads through the official CLI without login. If Kaggle changes that behavior, run `kaggle auth login` once.

## Run the application

```bash
source .venv/bin/activate
clickless
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

In Chrome or Edge, select **Speak request** to dictate the vehicle request. The browser asks for microphone permission the first time. Typed input always remains available.

## System configurations

The interface exposes the five proposal configurations:

- A: rule parsing, BM25 retrieval, and VADER aspect sentiment
- B: flat all-MiniLM-L6-v2 retrieval without graph paths
- C: full ontology graph retrieval with Qwen aspect extraction when the model is connected
- D: graph retrieval over raw reviews without aspect-assertion nodes
- E: graph retrieval with sentiment weights removed from ranking

Hard price, mileage, year, body-style, and drivetrain filters remain deterministic in every configuration.

## Live data refresh

After adding MarketCheck, Auto.dev, or SerpApi credentials to `.env`:

```bash
source .venv/bin/activate
python scripts/refresh_retail.py
```

MarketCheck is the primary live vehicle source when configured and Auto.dev is its fallback. SerpApi supplies live aftermarket offers across Google Shopping merchants. Live records are timestamped and labeled `Live`; research records are labeled `Frozen snapshot`. OpinRank remains the vehicle-owner review source because shopping reviews describe products rather than long-term vehicle ownership. Direct eBay and Best Buy refreshes run only when their credentials are present.

For a quick pre-demo refresh that avoids consuming many API calls:

```bash
python scripts/refresh_retail.py --per-vehicle 1 --max-vehicles 3
```

One-page Playwright capture with JSON-LD extraction is available for permitted research use:

```bash
clickless-crawl ebay "2009 Subaru Forester"
```

Import pages captured with `clickless-crawl` into the catalog (they appear as `Live capture` records):

```bash
clickless-crawl walmart "2009 Toyota RAV4 battery"
python scripts/import_captures.py
```

During the demo, each recommended car has a **Check live listings now** button. It queries MarketCheck, Auto.dev, and eBay (whichever keys are set) for that exact year, make, and model and shows the results with a timestamp.

## Neo4j and RDF

Start Neo4j and load the catalog:

```bash
docker compose up -d neo4j
python scripts/load_neo4j.py --clear
```

Then set `CLICKLESS_GRAPH_BACKEND=neo4j` in `.env` and restart the app. If Neo4j is unavailable, retrieval falls back to the same ontology graph in memory and labels the returned paths as `memory_fallback`.

Export and validate the Schema.org/PROV-O graph:

```bash
python scripts/export_rdf.py
```

The Turtle export is written to `data/processed/clickless.ttl` and checked against `data/shapes.ttl`.

## Tests

```bash
pip install -e '.[dev]'
PYTHONPATH=src pytest -q
```

## Evaluation

Run the 15-development/35-test scenario benchmark:

```bash
clickless-evaluate --configurations ABCDE
```

Add `--use-model` after the ASU key is configured to evaluate Qwen parsing and summaries.

The evaluation writes:

- `system_results.jsonl`
- `summary.json`
- `human_judgments.csv`
- `absa_annotations.csv`

The team must complete the blinded scores and review labels before calculating human-evaluation metrics.

After both judges fill `human_judgments.csv`, calculate Precision@5, nDCG@5, Cohen's kappa, and paired bootstrap confidence intervals:

```bash
clickless-analyze-evaluation
```

The runner also scores 10 unanswerable scenarios (abstention accuracy), gold-slot constraint satisfaction, and cohort retention. Add `--use-model` to `clickless-analyze-evaluation` to score Qwen ABSA against the annotated test sentences.

Generate the 50-query slot-parser benchmark, a 100-link graph audit sheet, SHACL status, and the web-extraction audit from captured retail pages:

```bash
clickless-audit-components
```

The graph and web precision fields remain pending until a team member fills the gold columns. Existing audit labels are preserved unless `--overwrite` is passed.
