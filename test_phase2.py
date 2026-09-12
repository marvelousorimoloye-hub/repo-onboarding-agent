"""
Phase 2 smoke test.

Indexes an already-registered repo (from a Phase 1 test run) and runs a
sample search query, printing the top results with similarity scores so
you can eyeball whether retrieval looks right.

Usage:
    python test_phase2.py <repo_name> "<search query>"

Example:
    python test_phase2.py wax-co-agentic-infrastructure "how does the n8n webhook retry logic work"
"""
import sys

from ingestion.repo_registry import RepoRegistry
from ingestion.semantic_indexer import SemanticIndexer


def main():
    if len(sys.argv) < 3:
        print('Usage: python test_phase2.py <repo_name> "<search query>"')
        sys.exit(1)

    repo_name = sys.argv[1]
    query = sys.argv[2]

    registry = RepoRegistry()
    record = registry.get_repo_by_name(repo_name)
    if record is None:
        print(f"No repo registered with name '{repo_name}'. Register it first "
              f"(e.g. via test_phase1.py) before indexing.")
        sys.exit(1)

    indexer = SemanticIndexer()

    print(f"Indexing {repo_name} ({record.id})...")
    indexer.index_repo(record.id)
    print("Done.")

    print(f"\nSearching for: {query!r}\n")
    results = indexer.search(query, repo_ids=[record.id], top_k=5)

    if not results:
        print("No results — index may be empty (check the repo actually has "
              "code/doc/commit content to chunk).")
        return

    for i, r in enumerate(results, 1):
        print(f"--- Result {i} (similarity={r['similarity']:.3f}, type={r['source_type']}) ---")
        if r["file_path"]:
            print(f"  file: {r['file_path']}")
        if r["graph_node_id"]:
            qualified_name = r["metadata"].get("qualified_name", "")
            print(f"  precise chunk — graph_node_id={r['graph_node_id']} ({qualified_name})")
        elif "start_line" in r["metadata"]:
            print(f"  windowed chunk — lines {r['metadata']['start_line']}-{r['metadata']['end_line']}")
        preview = r["content"][:300].replace("\n", " ")
        print(f"  {preview}{'...' if len(r['content']) > 300 else ''}\n")


if __name__ == "__main__":
    main()