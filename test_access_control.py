"""
Phase 3 access-control test.

Restricts a test user to ONE of two repos via repo_access, then runs the
same query as two different users — one restricted, one unrestricted —
to confirm RetrievalAgent actually filters results down to the allowlist
rather than just accepting whatever repo_ids the Router suggested.

Usage:
    python test_access_control.py <repo_name_allowed> <repo_name_other> "<query>"

Example:
    python test_access_control.py swagger-petstore PetStoreAPI "pet"
"""
import sys

from ingestion.repo_registry import RepoRegistry
from agents.retrieval_agent import RetrievalAgent
from db.connection import connection

RESTRICTED_USER = "restricted-test-user"
UNRESTRICTED_USER = "unrestricted-test-user"


def grant_access(user_id: str, repo_id: str) -> None:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO repo_access (user_id, repo_id)
                VALUES (%s, %s)
                ON CONFLICT (user_id, repo_id) DO NOTHING
                """,
                (user_id, repo_id),
            )


def clear_access(user_id: str) -> None:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM repo_access WHERE user_id = %s", (user_id,))


def main():
    if len(sys.argv) < 4:
        print('Usage: python test_access_control.py <repo_name_allowed> <repo_name_other> "<query>"')
        sys.exit(1)

    repo_name_allowed, repo_name_other, query = sys.argv[1], sys.argv[2], sys.argv[3]

    registry = RepoRegistry()
    allowed = registry.get_repo_by_name(repo_name_allowed)
    other = registry.get_repo_by_name(repo_name_other)
    if allowed is None or other is None:
        print("Both repos need to already be registered and indexed.")
        sys.exit(1)

    # Clean slate, then grant the restricted user access to ONLY `allowed`.
    clear_access(RESTRICTED_USER)
    clear_access(UNRESTRICTED_USER)  # stays empty — that's the point
    grant_access(RESTRICTED_USER, allowed.id)

    retrieval = RetrievalAgent()
    both_repo_ids = [allowed.id, other.id]

    print(f"Query: {query!r}, searching both {repo_name_allowed} and {repo_name_other}\n")

    print(f"--- As '{RESTRICTED_USER}' (allowlisted to {repo_name_allowed} only) ---")
    restricted_results = retrieval.retrieve(query, both_repo_ids, RESTRICTED_USER)
    repos_seen_restricted = {r["repo_id"] for r in restricted_results}
    print(f"{len(restricted_results)} chunks, from repo_id(s): {repos_seen_restricted}")
    if repos_seen_restricted - {allowed.id}:
        print(f"  FAIL — results included repo(s) outside the allowlist: {repos_seen_restricted - {allowed.id}}")
    elif not restricted_results:
        print("  (0 results — can't confirm filtering worked from this alone; "
              "try a query more likely to match the allowed repo)")
    else:
        print(f"  PASS — every result came from {repo_name_allowed}, as expected")

    print(f"\n--- As '{UNRESTRICTED_USER}' (no repo_access rows — unrestricted default) ---")
    unrestricted_results = retrieval.retrieve(query, both_repo_ids, UNRESTRICTED_USER)
    repos_seen_unrestricted = {r["repo_id"] for r in unrestricted_results}
    print(f"{len(unrestricted_results)} chunks, from repo_id(s): {repos_seen_unrestricted}")
    if len(repos_seen_unrestricted) > 1:
        print(f"  PASS — results spanned both repos, confirming the restricted "
              f"user's narrower results above were due to the allowlist, not "
              f"some other reason both users happened to get 1 repo")
    else:
        print(f"  Only one repo appeared here too — inconclusive on its own; "
              f"the {RESTRICTED_USER} result above being narrower is still "
              f"the meaningful comparison if this repo's content just doesn't "
              f"match well")


if __name__ == "__main__":
    main()