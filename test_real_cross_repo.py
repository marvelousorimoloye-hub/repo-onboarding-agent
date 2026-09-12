"""
Cross-Repo Linker test against real forked repos (not fixtures).

Registers swagger-api/swagger-petstore (has a committed openapi.yaml —
CONFIRMED-tier target) and DionisIno/PetStoreAPI (Python requests-based
test suite calling the Petstore API — the caller), then runs
CrossRepoLinker.link() SCOPED to just these two repos (not every active
repo in the registry — earlier runs left old fixture repos registered,
which polluted results when link() was called with no scope) and prints
whatever it finds.

Edit REPOS below to point at your forks (replace <your-username>).

Usage:
    python test_real_cross_repo.py
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from ingestion.repo_registry import RepoRegistry
from ingestion.cross_repo_linker import CrossRepoLinker
from db.connection import connection

REPOS = [
    # (name, git_url, default_branch)
    ("swagger-petstore", "https://github.com/marvelousorimoloye-hub/swagger-petstore.git", "master"),
    ("PetStoreAPI", "https://github.com/marvelousorimoloye-hub/PetStoreAPI.git", "main"),
]


def node_label(node_id: str) -> str:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT r.name, gn.qualified_name FROM graph_nodes gn
                JOIN repos r ON gn.repo_id = r.id
                WHERE gn.id = %s
                """,
                (node_id,),
            )
            row = cur.fetchone()
    return f"{row[0]}:{row[1]}" if row else f"<unknown:{node_id}>"


def summarize_repo(repo_id: str, repo_name: str):
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT node_type, count(*) FROM graph_nodes WHERE repo_id = %s GROUP BY node_type",
                (repo_id,),
            )
            counts = cur.fetchall()
            cur.execute(
                "SELECT file_path, metadata FROM graph_nodes "
                "WHERE repo_id = %s AND node_type = 'file' "
                "AND metadata::text LIKE '%%call_site_literals%%'",
                (repo_id,),
            )
            http_calls = cur.fetchall()
            cur.execute(
                "SELECT metadata FROM graph_nodes "
                "WHERE repo_id = %s AND qualified_name = '__mapping_errors__'",
                (repo_id,),
            )
            errors_row = cur.fetchone()
    print(f"  {repo_name}: " + ", ".join(f"{t}={c}" for t, c in counts) if counts else f"  {repo_name}: no nodes")
    for file_path, metadata in http_calls:
        print(f"    call-site literals in {file_path}: {metadata.get('call_site_literals', [])}")
    if errors_row is not None:
        skipped = errors_row[0].get("skipped_files", [])
        print(f"    {len(skipped)} file(s) skipped during mapping:")
        for entry in skipped:
            print(f"      {entry['file_path']}: {entry['error']}")


def main():
    registry = RepoRegistry()
    repo_ids = {}

    for repo_name, git_url, branch in REPOS:
        if "<your-username>" in git_url:
            print(f"Edit REPOS in this script — replace <your-username> for {repo_name}.")
            return

        existing = registry.get_repo_by_name(repo_name)
        if existing is not None:
            print(f"{repo_name} already registered — purging for a clean re-test...")
            registry.purge(existing.id)

        print(f"Registering {repo_name} from {git_url} (branch={branch})...")
        repo_id = registry.register(name=repo_name, git_url=git_url, default_branch=branch)
        repo_ids[repo_name] = repo_id

    print("\n=== Per-repo node summary ===")
    for repo_name, repo_id in repo_ids.items():
        summarize_repo(repo_id, repo_name)

    print("\nRunning CrossRepoLinker (scoped to just these two repos)...")
    linker = CrossRepoLinker()
    edges = linker.link(repo_ids=list(repo_ids.values()))

    print(f"\n=== {len(edges)} cross-repo edges found ===\n")
    by_confidence: dict[str, list] = {"confirmed": [], "inferred": []}
    for edge in edges:
        by_confidence.setdefault(edge.confidence, []).append(edge)

    for confidence in ("confirmed", "inferred"):
        print(f"--- {confidence.upper()} ({len(by_confidence[confidence])}) ---")
        for edge in by_confidence[confidence]:
            print(f"  [{edge.edge_type}] {node_label(edge.source_node_id)}  -->  {node_label(edge.target_node_id)}")
        print()

    if not edges:
        print("No edges found. Check the call-site literals printed above under "
              "'Per-repo node summary' — if PetStoreAPI shows none, or shows only "
              "a bare host with no path, the actual path-building pattern in that "
              "file isn't one we resolve yet. Paste that output back and we can "
              "dig into the actual pattern.")


if __name__ == "__main__":
    main()