"""
Postgres connection handling (pgvector for embeddings, plain tables for
graph metadata, and the LangGraph Postgres checkpointer share this same
database).

Phase 0 — fully implemented. Phase 2 addition: register_vector() on every
connection, so Python lists convert to/from pgvector's `vector` type
automatically instead of needing manual string formatting at every
call site that touches the embedding column.
"""
import psycopg2
import psycopg2.extras
from contextlib import contextmanager
from pgvector.psycopg2 import register_vector

from config.settings import DB


def get_connection():
    """Raw psycopg2 connection. Caller is responsible for closing it,
    or use `connection()` context manager below instead."""
    conn = psycopg2.connect(DB.dsn)
    psycopg2.extras.register_uuid()
    try:
        register_vector(conn)
    except psycopg2.ProgrammingError:
        # The `vector` extension isn't created yet (first-ever run, before
        # ensure_pgvector_extension() has executed) — safe to skip
        # registration this once, ensure_pgvector_extension()/run_schema()
        # will have created it before any embedding code actually runs.
        pass
    return conn


@contextmanager
def connection():
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def ensure_pgvector_extension() -> None:
    """Run once at startup / migration time."""
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")


def run_schema(schema_path: str = "db/schema.sql") -> None:
    """Applies db/schema.sql. Idempotent — schema.sql uses IF NOT EXISTS
    throughout."""
    with open(schema_path, "r") as f:
        schema_sql = f.read()
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(schema_sql)


if __name__ == "__main__":
    ensure_pgvector_extension()
    run_schema()
    print("Schema applied successfully.")