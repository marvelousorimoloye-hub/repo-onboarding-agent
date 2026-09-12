"""
Drift Detector — Phase 7 (SKELETON, not implemented).

TODO (Phase 7):
- Given a code diff/PR, compare against existing doc_sections claims for
  the changed repo, flag stale sentences.
- Fan out: check whether the change invalidates docs in dependent repos
  via cross-repo graph edges.
- Write results to `drift_flags` table.
"""
from config.settings import AgentRole


class DriftDetector:
    role = AgentRole.DRIFT_DETECTOR

    def check_pr(self, repo_id: str, commit_sha: str, diff: str) -> list[dict]:
        raise NotImplementedError("Phase 7 — Drift Detector not yet implemented")
