"""
Verifier / Grounding Agent — Phase 6 (SKELETON, not implemented).

TODO (Phase 6):
- Check each claim in a generated doc section against the actual code
  (read cited files/functions) before the doc is marked published.
- Reject ungrounded claims back to the Architecture Synthesizer with
  specifics on what failed verification.
"""
from config.settings import AgentRole


class VerifierAgent:
    role = AgentRole.VERIFIER

    def verify(self, doc_content: str, repo_id: str) -> dict:
        """Returns {"passed": bool, "failed_claims": list[str]}."""
        raise NotImplementedError("Phase 6 — Verifier Agent not yet implemented")
