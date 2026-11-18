"""Re-run the review aspect labeller on the saved catalog. No downloads needed.

Use this after updating clickless/data/aspects.py (for example the negation fix,
so "not really reliable" is labelled negative) instead of rebuilding all data.
Only rule-based ("lexicon") labels are replaced; Qwen labels are kept.

    python scripts/relabel_reviews.py            # update data/processed/catalog.json
    python scripts/relabel_reviews.py --dry-run  # only print what would change
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from clickless.config import settings
from clickless.data import load_catalog
from clickless.data.aspects import extract_aspects
from clickless.data.repository import validate_catalog


def relabel(catalog: dict) -> Counter:
    changes: Counter = Counter()
    for review in catalog["reviews"]:
        old = review.get("aspects") or []
        if any(item.get("method", "lexicon") != "lexicon" for item in old):
            changes["kept_model_labels"] += 1
            continue
        new = extract_aspects(str(review.get("text") or ""), review["id"])
        before = {(item["aspect"], item["span"]): item["polarity"] for item in old}
        after = {(item["aspect"], item["span"]): item["polarity"] for item in new}
        changes["labels_before"] += len(before)
        changes["labels_after"] += len(after)
        changes["polarity_changed"] += sum(
            1 for key, value in after.items() if key in before and before[key] != value
        )
        changes["labels_removed"] += len(before.keys() - after.keys())
        changes["labels_added"] += len(after.keys() - before.keys())
        review["aspects"] = new
    return changes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default=settings.catalog_path)
    args = parser.parse_args()
    catalog = load_catalog()
    changes = relabel(catalog)
    for key in sorted(changes):
        print(f"{key:20} {changes[key]}")
    if args.dry_run:
        return
    validate_catalog(catalog)
    Path(args.output).write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
