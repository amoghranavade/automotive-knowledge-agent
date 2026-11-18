import argparse
from pathlib import Path

from clickless.config import settings
from clickless.data import load_catalog
from clickless.graph.semantic import validate_rdf, write_rdf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=settings.processed_path / "clickless.ttl")
    parser.add_argument("--shapes", default=settings.root / "data" / "shapes.ttl")
    args = parser.parse_args()
    output = Path(args.output)
    write_rdf(load_catalog(), output)
    conforms, report = validate_rdf(output, Path(args.shapes))
    print(f"Wrote {output}")
    print(report)
    if not conforms:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
