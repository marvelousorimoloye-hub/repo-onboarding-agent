"""
Retrieval Agent — Phase 3.

Pulls relevant chunks via the Semantic Indexer (Phase 2), scoped to
whichever repo(s) the Router decided on and filtered by per-user access
(see db/schema.sql's repo_access table for the access model this enforces).
"""
from config.settings import AgentRole
from ingestion.semantic_indexer import SemanticIndexer
from db.connection import connection


class RetrievalAgent:
    # No LLM call is currently made in retrieve() — SemanticIndexer.search()
    # already does embedding + similarity search directly. This role is
    # kept for a plausible future addition (query rewriting/expansion
    # before searching, or reranking results) rather than removed, since
    # the plan explicitly assigns Retrieval to Groq for exactly that kind
    # of fast, high-frequency use.
    role = AgentRole.RETRIEVAL

    def __init__(self):
        self._indexer = SemanticIndexer()

    def retrieve(self, query: str, repo_ids: list[str], user_id: str, top_k: int = 8) -> list[dict]:
        allowed_repo_ids = self._filter_by_access(repo_ids, user_id)
        if not allowed_repo_ids:
            return []
        return self._indexer.search(query, repo_ids=allowed_repo_ids, top_k=top_k)

    def _filter_by_access(self, repo_ids: list[str], user_id: str) -> list[str]:
        """Access model (see db/schema.sql's repo_access table docstring
        for the full reasoning): a user_id with ZERO rows in repo_access
        has unrestricted access to whatever repo_ids the Router already
        scoped to — the sensible dev/test default, since no real
        permission system exists yet. A user_id with ANY rows is treated
        as an allowlist — only those specific repos, even if the Router
        suggested others."""
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT repo_id FROM repo_access WHERE user_id = %s",
                    (user_id,),
                )
                rows = cur.fetchall()
        if not rows:
            return repo_ids
        allowed = {str(r[0]) for r in rows}
        return [rid for rid in repo_ids if rid in allowed]