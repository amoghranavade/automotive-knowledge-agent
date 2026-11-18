"""The evaluation summary keeps metrics and drops listing details."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("summarize_evaluation", ROOT / "scripts" / "summarize_evaluation.py")
summarize_evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summarize_evaluation)


def _record(scenario, cfg, answerable, ids, mode):
    return {"scenario_id": scenario, "split": "test" if answerable else "unanswerable", "query": f"query {scenario}",
            "answerable": answerable, "configuration": cfg, "citation_coverage": 1.0, "constraint_satisfied": mode == "exact",
            "shown_cars_meet_limits": 1.0 if answerable else None, "abstained_correctly": None if answerable else True,
            "candidate_ids": ids,
            "result": {"match_mode": mode, "model_mode": "rules",
                       "candidates": [{"id": i, "location": "portland, or", "evidence": [{"url": "https://portland.craigslist.org/x"}]} for i in ids]}}


def test_summary_report_has_metrics_and_no_listing_details(tmp_path):
    run = tmp_path / "run"; run.mkdir()
    records = [_record("s1", "C", True, ["car-a", "car-b", "car-c"], "exact"),
               _record("s1", "E", True, ["car-b", "car-a", "car-c"], "exact"),
               _record("u1", "C", False, ["car-d"], "closest_alternatives"),
               _record("u1", "E", False, ["car-d"], "closest_alternatives")]
    (run / "system_results.jsonl").write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    metrics = {"runs": 2, "exact_match_rate": 1.0, "shown_cars_meet_limits": 1.0, "abstention_accuracy": 1.0,
               "citation_coverage": 1.0, "mean_cohort_retention": 0.5, "median_latency_seconds": 0.1, "p95_latency_seconds": 0.2}
    (run / "summary.json").write_text(json.dumps({"runs": 4, "configurations": ["C", "E"],
                                                  "by_configuration": {"C": metrics, "E": metrics}}), encoding="utf-8")
    out = tmp_path / "docs"
    report = summarize_evaluation.summarize(run, out, "test run", None).read_text(encoding="utf-8")
    assert "1 of 1" in report                      # exact matches and impossible requests
    assert "| E | C without sentiment in the score | 100% | 100% | 100% | 0 of 1 |" in report
    everything = "".join(p.read_text(encoding="utf-8") for p in out.iterdir())
    assert "craigslist" not in everything and "portland" not in everything
    assert "car-a | car-b | car-c" in (out / "per_request_C.csv").read_text(encoding="utf-8")
