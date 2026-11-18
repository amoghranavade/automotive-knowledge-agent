import argparse
import os

from neo4j import GraphDatabase

from clickless.data import load_catalog
from clickless.graph import load_into_neo4j


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "clickless-local")

    with GraphDatabase.driver(uri, auth=(user, password)) as driver:
        counts = load_into_neo4j(driver, load_catalog(), clear=args.clear)
    print(f"Loaded {counts['nodes']} nodes and {counts['edges']} relationships.")


if __name__ == "__main__":
    main()
