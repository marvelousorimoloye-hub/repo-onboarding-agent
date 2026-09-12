"""
Answer Synthesizer — Phase 3.

Composes a cited answer from whatever the Retrieval Agent found. Grounded
only in the provided chunks — explicitly instructed not to invent details,
and to say what's missing rather than guess when context is incomplete.

Citation scope, stated plainly: [N] markers in the answer text resolve to
real file:line locations (or a source description for non-code chunks like
commits/PRs) via the returned `sources` list — closing the gap between
Phase 3's original plan ("citations to actual file paths/line ranges") and
an earlier version of this file that only exposed opaque, unresolved [N]
index numbers. What this does NOT do: verify that a given [N] citation is
actually supported by that chunk's content — that's the Verifier/Grounding
Agent's job (Phase 6), which doesn't exist yet. These citations are the
model's self-reported attribution, not an audited one.
"""
from config.settings import AgentRole
from models.llm_clients import get_client_for_role


class AnswerSynthesizer:
    role = AgentRole.ANSWER_SYNTHESIZER

    def synthesize(self, question: str, retrieved_chunks: list[dict]) -> dict:
        """Returns {"answer": str, "sources": [{"index": int, "citation": str,
        "similarity": float}, ...]}. `answer` also has a rendered "Sources:"
        section appended, so printing it directly (as a quick console demo
        would) still shows resolved citations without the caller needing to
        do anything with the structured `sources` list — but a real UI
        should use `sources` directly rather than re-parsing the text."""
        if not retrieved_chunks:
            return {
                "answer": (
                    "I couldn't find anything relevant to answer that — either "
                    "nothing matching was indexed, or access to the relevant "
                    "repo(s) may be restricted for this user."
                ),
                "sources": [],
            }

        context_blocks = []
        sources = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            citation = self._format_citation(chunk)
            context_blocks.append(f"[{i}] {citation}:\n{chunk['content']}")
            sources.append({
                "index": i,
                "citation": citation,
                "similarity": chunk.get("similarity"),
            })
        context_text = "\n\n".join(context_blocks)

        system_prompt = (
            "You are a codebase Q&A assistant. Answer the user's question "
            "using ONLY the provided context snippets — do not invent "
            "details not present in them. Cite which snippet(s) support "
            "each claim using their [N] numbers. If the context doesn't "
            "fully answer the question, say plainly what's missing rather "
            "than guessing or filling gaps from general knowledge."
        )
        user_prompt = f"Context:\n{context_text}\n\nQuestion: {question}"

        client = get_client_for_role(self.role)
        raw_answer = client.complete(system_prompt, user_prompt)

        sources_section = "\n".join(f"  [{s['index']}] {s['citation']}" for s in sources)
        full_answer = f"{raw_answer}\n\nSources:\n{sources_section}"

        return {"answer": full_answer, "sources": sources}

    @staticmethod
    def _format_citation(chunk: dict) -> str:
        """Resolves a chunk to a real, human-usable location rather than
        just its source_type — file_path:start_line-end_line when both are
        available (precise code/api_spec chunks), file_path alone when
        line info isn't (windowed chunks, docs), or a source-specific
        description for chunks with no file at all (commit sha, PR number)."""
        file_path = chunk.get("file_path")
        metadata = chunk.get("metadata") or {}
        start_line = metadata.get("start_line")
        end_line = metadata.get("end_line")

        if file_path and start_line and end_line:
            qualified_name = metadata.get("qualified_name")
            suffix = f" ({qualified_name})" if qualified_name else ""
            return f"{file_path}:{start_line}-{end_line}{suffix}"
        if file_path:
            return file_path

        source_type = chunk.get("source_type", "unknown")
        if source_type == "commit" and metadata.get("commit_sha"):
            return f"commit {metadata['commit_sha'][:8]}"
        if source_type == "pr_description" and metadata.get("pr_number"):
            return f"PR #{metadata['pr_number']}"
        return f"({source_type})"