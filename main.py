"""
Entrypoint — SKELETON.

Currently only wires up Phase 0 (config, DB, model clients). Will call into
orchestration/graph.py once Phase 8 is implemented.
"""
from dotenv import load_dotenv
load_dotenv()  # This must run first

from config.settings import DB
from db.connection import ensure_pgvector_extension, run_schema


def bootstrap() -> None:
    """Phase 0: verify DB is reachable and schema is applied."""
    ensure_pgvector_extension()
    run_schema()
    print(f"Connected to {DB.database} at {DB.host}:{DB.port}. Schema applied.")


if __name__ == "__main__":
    bootstrap()
    # TODO: once Phase 8 lands, wire in orchestration.graph.build_query_graph()
    # etc. and expose a CLI / API entrypoint here.
