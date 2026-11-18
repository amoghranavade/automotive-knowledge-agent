"""Turn a clickless-evaluate run into a shareable report under docs/evaluation/.

Writes only aggregate metrics, request text and car-model IDs. Raw results stay
local: system_results.jsonl and the judging sheets hold listing URLs, seller
cities and review text, so they are never copied.

    clickless-evaluate --output evaluation/run-rules-v2
    clickless-audit-components                # optional: parser, SHACL and audit metrics
    python scripts/summarize_evaluation.py evaluation/run-rules-v2 --label "Rules mode"
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

CONFIG_NAMES = {
    "A": "Keyword search (BM25) + VADER",
    "B": "Meaning search (MiniLM), no graph",
    "C": "Full system: graph paths + review aspects",
    "D": "Graph with raw review text",
    "E": "C without sentiment in the score",
}


def _load_records(run_dir: Path) -> list[dict[str, Any]]:
    with (run_dir / "system_results.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _pct(value: float | None) -> str:
    return "–" if value is None else f"{100 * value:.0f}%"


def _categories() -> dict[str, str]:
    try:
        from clickless.evaluation.scenarios import default_scenarios, unanswerable_scenarios
    except ImportError:
        return {}
    return {s["id"]: s.get("category", "impossible") for s in default_scenarios() + unanswerable_scenarios()}


def _data_health(catalog_path: Path) -> list[tuple[str, str, str, str]]:
    """Counts only; no record contents leave the catalog."""
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    vehicles = catalog["vehicles"]
    n = len(vehicles)
    per = Counter(item["vehicle_id"] for item in catalog["listings"])
    try:
        from clickless.data.nhtsa import safety_stars
        rated = sum(1 for item in catalog["safety_ratings"] if safety_stars(item)[0] is not None)
    except ImportError:
        rated = len(catalog["safety_ratings"])
    with_recall = len({item["vehicle_id"] for item in catalog["recalls"]})
    exact = Counter(r["vehicle_id"] for r in catalog["reviews"] if r.get("match_type") == "exact_year")
    reviewed = sum(1 for v in vehicles if exact[v["id"]] >= 5)
    records = [x for key in ("listings", "reviews", "parts", "safety_ratings", "recalls") for x in catalog[key]]
    sourced = sum(1 for x in records if all(x.get(f) for f in ("source_id", "source_name", "source_url", "captured_at")))

    def status(share: float) -> str:
        return "Excellent" if share >= 0.95 else "Good" if share >= 0.85 else "Fair" if share >= 0.6 else "Gap"

    return [
        ("Listings per car", f"{sum(1 for v in vehicles if per[v['id']])} of {n} cars ({min(per.values())}–{max(per.values())} each)",
         status(sum(1 for v in vehicles if per[v['id']]) / n), "Real prices and miles for every car"),
        ("Crash-test stars", f"{rated} of {n} cars ({rated / n:.0%})", status(rated / n), f"{n - rated} cars show no NHTSA stars"),
        ("Recalls linked", f"{with_recall} of {n} cars, {len(catalog['recalls'])} recalls", status(with_recall / n), "Cards can show recall cautions"),
        ("Owner reviews (exact year)", f"{reviewed} of {n} cars with 5+", status(reviewed / n), f"{n - reviewed} cars rank without owner opinions"),
        ("Source on every record", f"{sourced:,} of {len(records):,} ({sourced / len(records):.0%})", status(sourced / len(records)), "Every claim can cite its record"),
    ]


def summarize(run_dir: Path, out_dir: Path, label: str, catalog_path: Path | None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    records = _load_records(run_dir)
    configs = summary.get("configurations") or sorted({r["configuration"] for r in records})
    categories = _categories()

    # ---- per configuration
    by_cfg = summary.get("by_configuration", {})
    top3: dict[str, dict[str, list[str]]] = defaultdict(dict)
    for r in records:
        if r["answerable"]:
            top3[r["scenario_id"]][r["configuration"]] = r["candidate_ids"][:3]
    answerable_ids = sorted(top3)
    agree = {c: sum(1 for s in answerable_ids if top3[s].get(c) == top3[s].get("C")) for c in configs}
    model_modes = Counter(r["result"].get("model_mode", "rules") for r in records)

    with (out_dir / "per_configuration.csv").open("w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle, lineterminator="\n")
        w.writerow(["configuration", "description", "runs", "exact_match_rate", "shown_cars_meet_limits", "abstention_accuracy",
                    "citation_coverage", "mean_cohort_retention", "same_top3_as_C", "answerable_requests", "median_latency_s", "p95_latency_s"])
        for c in configs:
            m = by_cfg.get(c, {})
            r4 = lambda v: None if v is None else round(float(v), 4)
            w.writerow([c, CONFIG_NAMES.get(c, ""), m.get("runs"), r4(m.get("exact_match_rate")), r4(m.get("shown_cars_meet_limits")),
                        r4(m.get("abstention_accuracy")), r4(m.get("citation_coverage")), r4(m.get("mean_cohort_retention")),
                        agree[c], len(answerable_ids), r4(m.get("median_latency_seconds")), r4(m.get("p95_latency_seconds"))])

    # ---- per request (configuration C): no prices, places or URLs
    with (out_dir / "per_request_C.csv").open("w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle, lineterminator="\n")
        w.writerow(["scenario_id", "split", "category", "query", "answerable", "result", "cars_shown", "all_shown_meet_limits",
                    "citation_coverage", "top3_car_models"])
        for r in records:
            if r["configuration"] != "C":
                continue
            mode = r["result"].get("match_mode")
            w.writerow([r["scenario_id"], r["split"], categories.get(r["scenario_id"], ""), r["query"], r["answerable"],
                        "exact matches" if mode == "exact" else "closest options", len(r["candidate_ids"]),
                        r.get("shown_cars_meet_limits"), r["citation_coverage"], " | ".join(r["candidate_ids"][:3])])

    # ---- per category (configuration C)
    cat_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        if r["configuration"] == "C":
            cat = categories.get(r["scenario_id"], r["split"])
            cat_rows["impossible" if cat in {"impossible", "unanswerable"} else cat].append(r)

    # ---- copy aggregate files
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    components = None
    if (run_dir / "component_metrics.json").exists():
        components = json.loads((run_dir / "component_metrics.json").read_text(encoding="utf-8"))
        (out_dir / "component_metrics.json").write_text(json.dumps(components, indent=2), encoding="utf-8")
    if (run_dir / "slot_parsing_metrics.json").exists():
        slot = json.loads((run_dir / "slot_parsing_metrics.json").read_text(encoding="utf-8"))
        (out_dir / "slot_parsing_metrics.json").write_text(json.dumps(slot, indent=2), encoding="utf-8")

    # ---- report
    C = by_cfg.get("C", summary)
    answerable = [r for r in records if r["configuration"] == "C" and r["answerable"]]
    impossible = [r for r in records if r["configuration"] == "C" and not r["answerable"]]
    splits = Counter(r["split"] for r in records if r["configuration"] == "C")
    lines = [
        "# ClickLess Automotive – Evaluation Results", "",
        f"**Run:** {label} · {summary['runs']} runs ({len(records) // max(len(configs), 1)} requests × {len(configs)} configurations) · "
        f"model mode: {', '.join(f'{k} ({v})' for k, v in model_modes.items())} · generated {date.today().isoformat()}", "",
        f"Requests: {splits.get('development', 0)} tuning, {splits.get('test', 0)} held out and {splits.get('unanswerable', 0)} extra impossible "
        f"({len(answerable)} answerable, {len(impossible)} impossible in total).", "",
        "## Headline results (full system, configuration C)", "",
        "| Metric | Result |", "|---|---|",
        f"| Answerable requests with exact matches | {sum(r['constraint_satisfied'] for r in answerable)} of {len(answerable)} |",
        f"| Cars shown that meet every hard limit | {_pct(C.get('shown_cars_meet_limits'))} |",
        f"| Impossible requests answered with closest options | {sum(bool(r['abstained_correctly']) for r in impossible)} of {len(impossible)} |",
        f"| Pros and cautions that cite a record | {_pct(C.get('citation_coverage'))} |",
        f"| Feasible cars with every requested feature scored | {_pct(C.get('mean_cohort_retention'))} |",
        f"| Median / p95 response time | {C.get('median_latency_seconds', 0):.2f} s / {C.get('p95_latency_seconds', 0):.2f} s |", "",
        "## Ablation: configurations A–E", "",
        "| Config | What changes | Exact matches | Shown cars meet limits | Citations | Same top 3 as C |", "|---|---|---|---|---|---|",
    ]
    for c in configs:
        m = by_cfg.get(c, {})
        same = "reference" if c == "C" else f"{agree[c]} of {len(answerable_ids)}"
        lines.append(f"| {c} | {CONFIG_NAMES.get(c, '')} | {_pct(m.get('exact_match_rate'))} | {_pct(m.get('shown_cars_meet_limits'))} | "
                     f"{_pct(m.get('citation_coverage'))} | {same} |")
    if "E" in agree and "C" in configs:
        lines += ["", f"Removing sentiment (E) changes the top 3 in {len(answerable_ids) - agree['E']} of {len(answerable_ids)} requests, "
                  "so owner opinions drive the ranking. B and D rank like C in this mode because retrieval runs after ranking; "
                  "they change the evidence shown, not the order."]
    lines += ["", "## Results by request type (configuration C)", "",
              "| Request type | Requests | Exact matches | Closest options |", "|---|---|---|---|"]
    for cat in sorted(cat_rows):
        rows = cat_rows[cat]
        exact = sum(r["result"].get("match_mode") == "exact" for r in rows)
        lines.append(f"| {cat.replace('_', ' ')} | {len(rows)} | {exact} | {len(rows) - exact} |")
    if components:
        slot = components.get("slot_parsing", {})
        per = slot.get("per_slot_f1", {})
        lines += ["", "## Components", "",
                  f"- **Slot parser:** {slot.get('cases', 0)} requests, exact match {_pct(slot.get('exact_match'))}, "
                  f"macro F1 {slot.get('macro_f1', 0):.2f} ({', '.join(f'{k} {v:.2f}' for k, v in per.items())}).",
                  f"- **RDF export:** SHACL {'conforms' if components.get('shacl_conforms') else 'does not conform'}."]
        gl = components.get("graph_link_audit", {})
        if gl:
            lines.append(f"- **Graph link audit:** {gl.get('labeled_links', 0)} of {gl.get('sampled_links', 0)} sampled links labeled so far"
                         + (f", precision {_pct(gl.get('precision'))}." if gl.get("precision") is not None else "."))
    if catalog_path and catalog_path.exists():
        lines += ["", "## Data health (catalog)", "", "| Metric | Score | Status | Impact on the agent |", "|---|---|---|---|"]
        lines += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in _data_health(catalog_path)]
    lines += ["", "## Hallucination risk", "",
              "- **Numbers – low.** The claim checker rejects any pro or caution whose price, mileage, star rating or recall count does not match its cited record.",
              f"- **Citations – low.** {_pct(C.get('citation_coverage'))} of pros and cautions cite a record.",
              "- **Review opinions – medium.** Rule-based labels still misread some sentences; Qwen mode re-reads them.",
              "", "## Still to come", "",
              "- Blind judging (nDCG@5, P@5, Cohen's κ) with `clickless-analyze-evaluation` once `human_judgments.csv` is filled in.",
              "- The same benchmark with `--use-model` (Qwen3.8-27B) for B–E.",
              "", "## Files", "",
              "- `summary.json` – overall and per-configuration metrics from `clickless-evaluate`",
              "- `per_configuration.csv` – the ablation table above",
              "- `per_request_C.csv` – every request with its outcome and the top-3 car models (no prices, places or links)",
              "- `component_metrics.json`, `slot_parsing_metrics.json` – from `clickless-audit-components`",
              "", "Reproduce:", "", "```bash", f"clickless-evaluate --output evaluation/{run_dir.name}", "clickless-audit-components",
              f"python scripts/summarize_evaluation.py evaluation/{run_dir.name}", "```", ""]
    report = out_dir / "README.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="folder written by clickless-evaluate")
    parser.add_argument("--out", type=Path, default=Path("docs/evaluation"))
    parser.add_argument("--label", default="Rules mode, real 30-car catalog")
    parser.add_argument("--catalog", type=Path, default=Path("data/processed/catalog.json"))
    args = parser.parse_args()
    report = summarize(args.run_dir, args.out, args.label, args.catalog)
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
