"""
LangGraph orchestration — Phase 8 (SKELETON, not implemented).

TODO (Phase 8):
- Wire Router -> Retrieval Agent -> (Trace Agent) -> Answer Synthesizer
  into a LangGraph StateGraph for the query-answering path.
- Wire Architecture Synthesizer -> Verifier Agent into a separate graph
  for the doc-generation path.
- Wire Drift Detector as a standalone webhook-triggered graph.
- Use PostgresSaver (langgraph-checkpoint-postgres) against the same DB
  as db/connection.py for checkpointing, so long/cross-repo traces are
  resumable.
"""


def build_query_graph():
    raise NotImplementedError("Phase 8 — orchestration not yet implemented")


def build_doc_generation_graph():
    raise NotImplementedError("Phase 8 — orchestration not yet implemented")


def build_drift_detection_graph():
    raise NotImplementedError("Phase 8 — orchestration not yet implemented")
