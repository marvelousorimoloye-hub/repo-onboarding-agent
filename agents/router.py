"""
Router — Phase 3.

Classifies an incoming question and figures out which registered repo(s)
it's scoped to. Single-hop only at this stage — no cross-file/cross-repo
tracing (that's the Trace Agent, Phase 4).
"""
import json

from config.settings import AgentRole
from models.llm_clients import get_client_for_role
from ingestion.repo_registry import RepoRegistry


class Router:
    role = AgentRole.ROUTER

    def __init__(self):
        self._registry = RepoRegistry()

    def route(self, question: str, user_id: str) -> dict:
        """Returns:
        {
            "question_type": "architecture_overview" | "specific_trace" | "why_historical" | "general",
            "repo_ids": [str, ...],       # best-effort scope to search
            "needs_clarification": bool,  # True if scope couldn't be confidently narrowed
            "clarification_message": str | None,
        }

        Ambiguity handling: if the model can't confidently narrow the
        question to specific repo(s) — e.g. a generic term like "auth"
        with 3 registered repos and nothing else narrowing it down —
        needs_clarification is True, but repo_ids STILL defaults to every
        active repo. A caller with no human in the loop (a test script, a
        non-interactive pipeline) gets a usable best-effort answer rather
        than nothing; a caller that DOES have a human in the loop can
        check the flag and ask before searching. Matches the general
        principle: ambiguity is a reason to pick a sensible default and
        proceed, not a reason to block."""
        active_repos = self._registry.list_active_repos()
        if not active_repos:
            return {
                "question_type": "general",
                "repo_ids": [],
                "needs_clarification": False,
                "clarification_message": None,
            }
        if len(active_repos) == 1:
            # No real ambiguity possible with only one repo — skip the
            # LLM call entirely, cheap heuristic is enough.
            return {
                "question_type": self._classify_question_type_heuristic(question),
                "repo_ids": [active_repos[0].id],
                "needs_clarification": False,
                "clarification_message": None,
            }

        repo_list_text = "\n".join(f"- {r.name}" for r in active_repos)
        system_prompt = (
            "You are a routing component in a codebase Q&A system. Given a "
            "user question and a list of registered repositories, decide "
            "which repositories the question is most likely about.\n"
            "Respond ONLY with JSON, no other text, in this exact shape:\n"
            '{"question_type": "architecture_overview"|"specific_trace"|"why_historical"|"general", '
            '"repo_names": ["..."], "confident": true|false}\n'
            "Set confident=false if the question could plausibly apply to "
            "more than one repo and nothing in the question narrows it down."
        )
        user_prompt = f"Repositories:\n{repo_list_text}\n\nQuestion: {question}"

        client = get_client_for_role(self.role)
        raw_response = client.complete(system_prompt, user_prompt)
        parsed = self._safe_parse_json(raw_response)

        clarification_message = (
            "Which repository is this about? Available: "
            + ", ".join(r.name for r in active_repos)
        )

        if parsed is None:
            # Model didn't return valid JSON — degrade to "search
            # everything, flagged as needing clarification" rather than
            # crashing the whole pipeline over a formatting slip.
            return {
                "question_type": "general",
                "repo_ids": [r.id for r in active_repos],
                "needs_clarification": True,
                "clarification_message": clarification_message,
            }

        name_to_id = {r.name: r.id for r in active_repos}
        matched_ids = [name_to_id[n] for n in parsed.get("repo_names", []) if n in name_to_id]
        confident = bool(parsed.get("confident", False))

        if not matched_ids or not confident:
            return {
                "question_type": parsed.get("question_type", "general"),
                "repo_ids": matched_ids or [r.id for r in active_repos],
                "needs_clarification": True,
                "clarification_message": clarification_message,
            }

        return {
            "question_type": parsed.get("question_type", "general"),
            "repo_ids": matched_ids,
            "needs_clarification": False,
            "clarification_message": None,
        }

    @staticmethod
    def _safe_parse_json(raw: str) -> dict | None:
        # Models sometimes wrap JSON in ```json fences despite instructions
        # not to — strip those before parsing rather than failing on them.
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
        try:
            return json.loads(cleaned.strip())
        except (json.JSONDecodeError, TypeError):
            return None

    @staticmethod
    def _classify_question_type_heuristic(question: str) -> str:
        # Used only in the single-repo fast path (skips the LLM call
        # entirely there) — a rough classification is still useful
        # downstream, not worth a wasted API call to get it precisely.
        lowered = question.lower()
        if any(w in lowered for w in ("why", "reason", "decided", "history")):
            return "why_historical"
        if any(w in lowered for w in ("architecture", "overview", "design")):
            return "architecture_overview"
        if any(w in lowered for w in ("how does", "trace", "flow", "call")):
            return "specific_trace"
        return "general"