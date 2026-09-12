"""
Debug: inspect graph_nodes/graph_edges directly to find exactly why
TraceAgent hit a dead end, rather than guessing.

Usage:
    python debug_trace_edges.py <repo_name> <file_path_substring>

Example:
    python debug_trace_edges.py wax-co-agentic-infrastructure tools.py
"""
import sys

from ingestion.repo_registry import RepoRegistry
from db.connection import connection


def main():
    if len(sys.argv) < 3:
        print("Usage: python debug_trace_edges.py <repo_name> <file_path_substring>")
        sys.exit(1)

    repo_name, file_substring = sys.argv[1], sys.argv[2]

    registry = RepoRegistry()
    record = registry.get_repo_by_name(repo_name)
    if record is None:
        print(f"No repo registered with name '{repo_name}'.")
        sys.exit(1)

    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, node_type, qualified_name, file_path, metadata "
                "FROM graph_nodes WHERE repo_id = %s AND file_path ILIKE %s",
                (record.id, f"%{file_substring}%"),
            )
            rows = cur.fetchall()

    print(f"Nodes with file_path matching '{file_substring}': {len(rows)}")
    file_node_id = None
    for r in rows:
        node_id, node_type, qualified_name, file_path, metadata = r
        print(f"  id={node_id} type={node_type} name={qualified_name} file_path={file_path!r}")
        if node_type == "file":
            file_node_id = str(node_id)
            raw_imports = (metadata or {}).get("raw_imports")
            print(f"    raw_imports metadata: {raw_imports}")

    if file_node_id is None:
        print("\nNo FILE node found matching that substring — stopping here.")
        return

    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT ge.edge_type, gn.qualified_name, gn.file_path "
                "FROM graph_edges ge JOIN graph_nodes gn ON ge.target_node_id = gn.id "
                "WHERE ge.source_node_id = %s",
                (file_node_id,),
            )
            edges = cur.fetchall()
    print(f"\nEdges FROM file node {file_node_id}: {len(edges)}")
    for e in edges:
        print(f"  {e[0]} -> {e[1]} ({e[2]!r})")

    # Also check whether resilience.py (or whatever the import target
    # should be) exists as a file node at all, and what its exact stored
    # file_path string looks like — a mismatch here (e.g. leftover
    # backslash, or a different directory than expected) would silently
    # break the exact-string match import resolution depends on.
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, file_path FROM graph_nodes "
                "WHERE repo_id = %s AND node_type = 'file' AND file_path ILIKE %s",
                (record.id, "%resilience%"),
            )
            res_rows = cur.fetchall()
    print(f"\nFile nodes matching 'resilience': {res_rows}")


if __name__ == "__main__":
    main()