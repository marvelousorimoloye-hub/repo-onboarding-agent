"""
Phase 2 cross-repo search test.

Indexes two already-registered repos and runs one search scoped to BOTH
via repo_ids, to confirm multi-repo scoping actually works (not just
single-repo, which test_phase2.py already covers).

Usage:
    python test_phase2_cross.py <repo_name_1> <repo_name_2> "<search query>"

Example:
    python test_phase2_cross.py swagger-petstore PetStoreAPI "how do I get a pet by id"
"""
import sys

from ingestion.repo_registry import RepoRegistry
from ingestion.semantic_indexer import SemanticIndexer


def main():
    if len(sys.argv) < 4:
        print('Usage: python test_phase2_cross.py <repo_name_1> <repo_name_2> "<search query>"')
        sys.exit(1)

    repo_name_1, repo_name_2, query = sys.argv[1], sys.argv[2], sys.argv[3]

    registry = RepoRegistry()
    records = []
    for name in (repo_name_1, repo_name_2):
        record = registry.get_repo_by_name(name)
        if record is None:
            print(f"No repo registered with name '{name}'. Register it first.")
            sys.exit(1)
        records.append(record)

    indexer = SemanticIndexer()
    for record in records:
        print(f"Indexing {record.name} ({record.id})...")
        indexer.index_repo(record.id)
        counts = indexer.get_chunk_counts(record.id)
        print(f"  chunk counts: {counts if counts else '(none — indexing produced nothing at all)'}")
    print("Done.\n")

    print(f"Searching BOTH repos for: {query!r}\n")
    results = indexer.search(query, repo_ids=[r.id for r in records], top_k=8)

    if not results:
        print("No results.")
        return

    repo_names_by_id = {r.id: r.name for r in records}
    seen_repos = set()
    for i, r in enumerate(results, 1):
        repo_name = repo_names_by_id.get(r["repo_id"], r["repo_id"])
        seen_repos.add(repo_name)
        print(f"--- Result {i} (repo={repo_name}, similarity={r['similarity']:.3f}, type={r['source_type']}) ---")
        if r["file_path"]:
            print(f"  file: {r['file_path']}")
        preview = r["content"][:200].replace("\n", " ")
        print(f"  {preview}{'...' if len(r['content']) > 200 else ''}\n")

    print(f"Repos represented in results: {seen_repos}")
    if len(seen_repos) < 2:
        print("(Only one repo showed up — could be correct if the query genuinely "
              "only matches one repo's content, or could mean scoping isn't truly "
              "searching both. Worth trying a query you know should hit both.)")


if __name__ == "__main__":
    main()