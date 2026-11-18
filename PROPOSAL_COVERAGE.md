# Proposal coverage

This file separates implemented behavior from work that requires credentials, external services, or human judgment.

## Implemented and verified

| Proposal requirement | Current evidence |
|---|---|
| 30 make-model-year variants | `data/processed/catalog.json` contains 30 variants. |
| About 1,000 listings | 969 cleaned Craigslist/Kaggle listing records plus one validated Auto.dev live listing; MarketCheck and Auto.dev refresh connectors are implemented. |
| About 2,000 reviews | 2,014 OpinRank and Amazon Automotive review records. |
| NHTSA safety and recalls | 23 variants have matched crash-rating responses and the catalog contains 196 recall records; live recall refresh is available in the UI. |
| Classical parsing and retrieval | Regex/lexicon constraint parser, TF-IDF, and BM25 are implemented. |
| Classical sentiment baselines | Configuration A uses VADER aspect sentiment; the completed annotation workflow trains TF-IDF plus logistic regression. |
| Modern model pipeline | Qwen JSON request parsing, bounded review-aspect extraction, and grounded summaries are implemented with deterministic claim verification. The ASU-hosted `qwen3-30b-a3b` path completed a full browser recommendation flow. |
| Dense retrieval | all-MiniLM-L6-v2 retrieval is implemented for B and as the dense fallback for C-E. |
| Linked-data graph | Neo4j loader, optional Neo4j runtime traversal, in-memory fallback, RDF/Turtle export, Schema.org mappings, PROV-O provenance, and SHACL validation are implemented. |
| Multi-hop GraphRAG | Cross-source subsystem alignment, generation-lineage evidence, raw-review ablation, and related product/recall paths are implemented. |
| Evidence safeguards | Same-generation reviews are labeled indirect and excluded from ranking; unverified product fitment cannot be presented as fact. |
| A-E configurations | A uses rule/BM25/VADER; B uses flat MiniLM; C uses full GraphRAG and Qwen aspects when connected; D retains raw reviews with C's ranking; E removes sentiment weights. |
| Minimum demonstration | The UI accepts typed or Chrome speech input, asks no more than two questions, and renders three ranked candidates with review evidence, recall cautions, source links, and aspect bars. |
| System evaluation | 250 A-E runs over 15 development and 35 held-out scenarios completed. Constraint satisfaction and citation coverage are both 1.0; median latency is 1.516 seconds and p95 latency is 1.872 seconds on this machine. |
| Component evaluation | The 50-query slot benchmark completed with macro-F1 1.0. RDF/SHACL validation conforms. A 100-link graph audit sheet is prepared. |
| Human-evaluation design | Blinded top-10 relevance sheets, 100-claim citation audit, 250 five-point explanation-rating rows, and 300 ABSA sentences with 120 duplicate annotations and 180/60/60 splits are generated. |

## Requires external input

1. Run the full model-enabled A-E benchmark if model-specific aggregate metrics are required. The interactive ASU Qwen path is connected and verified.
2. Expand the current validated Auto.dev and SerpApi sample refresh if more live records are needed. Missing MarketCheck, eBay, Best Buy, Amazon, and Walmart credentials do not block the demo.
3. Capture and manually audit up to 100 permitted eBay, Amazon, Walmart, and Best Buy pages. `web_extraction_audit.csv` currently has zero captured pages because no live browser captures have been supplied.
4. Start Neo4j with `docker compose up -d neo4j`, load it with `python scripts/load_neo4j.py --clear`, set `CLICKLESS_GRAPH_BACKEND=neo4j`, and restart the app. Docker is not installed on the current machine, so the verified runtime uses the equivalent in-memory ontology graph.
5. Two team members must complete the blinded relevance, ABSA, citation, graph-link, and explanation-rating sheets. Human metrics remain blank by design; the software does not fabricate judgments.

## Completion commands

```bash
source .venv/bin/activate
python scripts/refresh_retail.py --per-vehicle 3
python scripts/load_neo4j.py --clear
clickless-evaluate --configurations ABCDE --use-model
clickless-audit-components
clickless-analyze-evaluation
clickless
```
