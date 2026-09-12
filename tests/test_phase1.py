"""
Phase 1 smoke test.

Registers a real repo (clones it, runs RepoMapper, persists the graph),
then prints a summary so you can eyeball whether the extraction looks
right before we trust it for anything downstream.

Usage:
    python test_phase1.py <git_url> [branch]

Example:
    python test_phase1.py https://github.com/<you>/wax-co-ai-demo.git main
"""
import sys
import json

from db.connection import connection
from ingestion.repo_registry import RepoRegistry


def summarize(repo_id: str, repo_name: str):
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT node_type, count(*) FROM graph_nodes WHERE repo_id = %s GROUP BY node_type",
                (repo_id,),
            )
            node_counts = cur.fetchall()

            cur.execute(
                "SELECT edge_type, count(*) FROM graph_edges ge "
                "JOIN graph_nodes gn ON ge.source_node_id = gn.id "
                "WHERE gn.repo_id = %s GROUP BY edge_type",
                (repo_id,),
            )
            edge_counts = cur.fetchall()

            cur.execute(
                "SELECT qualified_name, file_path FROM graph_nodes "
                "WHERE repo_id = %s AND node_type = 'function' LIMIT 20",
                (repo_id,),
            )
            sample_functions = cur.fetchall()

            cur.execute(
                "SELECT file_path, metadata FROM graph_nodes "
                "WHERE repo_id = %s AND node_type = 'file' "
                "AND metadata::text LIKE '%%call_site_literals%%'",
                (repo_id,),
            )
            http_call_files = cur.fetchall()

    print(f"\n=== Summary for {repo_name} ({repo_id}) ===")
    print("\nNode counts:")
    for node_type, count in node_counts:
        print(f"  {node_type}: {count}")

    print("\nEdge counts:")
    for edge_type, count in edge_counts:
        print(f"  {edge_type}: {count}")

    print(f"\nSample functions found (up to 20):")
    for qualified_name, file_path in sample_functions:
        print(f"  {qualified_name}  ({file_path})")

    print(f"\nFiles with detected HTTP call-site literals:")
    if not http_call_files:
        print("  (none — expected for a repo with no outbound HTTP calls)")
    for file_path, metadata in http_call_files:
        literals = metadata.get("call_site_literals", [])
        print(f"  {file_path}: {literals}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_phase1.py <git_url> [branch]")
        sys.exit(1)

    git_url = sys.argv[1]
    branch = sys.argv[2] if len(sys.argv) > 2 else "main"
    repo_name = git_url.rstrip("/").split("/")[-1].removesuffix(".git")

    registry = RepoRegistry()

    existing = registry.get_repo_by_name(repo_name)
    if existing is not None:
        print(f"{repo_name} is already registered (repo_id={existing.id}) — purging for a clean re-test...")
        registry.purge(existing.id)

    print(f"Registering {repo_name} from {git_url} (branch={branch})...")
    repo_id = registry.register(name=repo_name, git_url=git_url, default_branch=branch)
    print(f"Registered. repo_id={repo_id}")

    summarize(repo_id, repo_name)