# ClickLess Automotive – Evaluation Results

**Run:** Rules mode, real 30-car catalog · 300 runs (60 requests × 5 configurations) · model mode: rules (300) · generated 2026-11-18

Requests: 15 tuning, 35 held out and 10 extra impossible (47 answerable, 13 impossible in total).

## Headline results (full system, configuration C)

| Metric | Result |
|---|---|
| Answerable requests with exact matches | 47 of 47 |
| Cars shown that meet every hard limit | 100% |
| Impossible requests answered with closest options | 13 of 13 |
| Pros and cautions that cite a record | 100% |
| Feasible cars with every requested feature scored | 67% |
| Median / p95 response time | 0.45 s / 0.70 s |

## Ablation: configurations A–E

| Config | What changes | Exact matches | Shown cars meet limits | Citations | Same top 3 as C |
|---|---|---|---|---|---|
| A | Keyword search (BM25) + VADER | 100% | 100% | 100% | 36 of 47 |
| B | Meaning search (MiniLM), no graph | 100% | 100% | 100% | 47 of 47 |
| C | Full system: graph paths + review aspects | 100% | 100% | 100% | reference |
| D | Graph with raw review text | 100% | 100% | 100% | 47 of 47 |
| E | C without sentiment in the score | 100% | 100% | 100% | 14 of 47 |

Removing sentiment (E) changes the top 3 in 33 of 47 requests, so owner opinions drive the ranking. B and D rank like C in this mode because retrieval runs after ranking; they change the evidence shown, not the order.

## Results by request type (configuration C)

| Request type | Requests | Exact matches | Closest options |
|---|---|---|---|
| aftermarket maintenance | 7 | 7 | 0 |
| budget mileage | 9 | 9 | 0 |
| conflicting priorities | 6 | 6 | 0 |
| family cargo | 6 | 6 | 0 |
| impossible | 13 | 0 | 13 |
| model year | 6 | 6 | 0 |
| sparse evidence | 6 | 6 | 0 |
| winter awd | 7 | 7 | 0 |

## Components

- **Slot parser:** 50 requests, exact match 100%, macro F1 1.00 (max_price 1.00, max_mileage 1.00, min_year 1.00, body_style 1.00, drivetrain 1.00).
- **RDF export:** SHACL conforms.
- **Graph link audit:** 0 of 100 sampled links labeled so far.

## Data health (catalog)

| Metric | Score | Status | Impact on the agent |
|---|---|---|---|
| Listings per car | 30 of 30 cars (22–40 each) | Excellent | Real prices and miles for every car |
| Crash-test stars | 27 of 30 cars (90%) | Good | 3 cars show no NHTSA stars |
| Recalls linked | 30 of 30 cars, 228 recalls | Excellent | Cards can show recall cautions |
| Owner reviews (exact year) | 20 of 30 cars with 5+ | Fair | 10 cars rank without owner opinions |
| Source on every record | 3,540 of 3,540 (100%) | Excellent | Every claim can cite its record |

## Hallucination risk

- **Numbers – low.** The claim checker rejects any pro or caution whose price, mileage, star rating or recall count does not match its cited record.
- **Citations – low.** 100% of pros and cautions cite a record.
- **Review opinions – medium.** Rule-based labels still misread some sentences; Qwen mode re-reads them.

## Still to come

- Blind judging (nDCG@5, P@5, Cohen's κ) with `clickless-analyze-evaluation` once `human_judgments.csv` is filled in.
- The same benchmark with `--use-model` (Qwen3.8-27B) for B–E.

## Files

- `summary.json` – overall and per-configuration metrics from `clickless-evaluate`
- `per_configuration.csv` – the ablation table above
- `per_request_C.csv` – every request with its outcome and the top-3 car models (no prices, places or links)
- `component_metrics.json`, `slot_parsing_metrics.json` – from `clickless-audit-components`

Reproduce:

```bash
clickless-evaluate --output evaluation/run-rules-v2
clickless-audit-components
python scripts/summarize_evaluation.py evaluation/run-rules-v2
```
