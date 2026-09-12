"""
Architecture Synthesizer — Phase 5 (SKELETON, not implemented).

TODO (Phase 5):
- Walk each repo's dependency graph, generate per-module docs using
  docs_templates/module_doc_template.md.
- Generate the system-level doc using
  docs_templates/system_level_doc_template.md, from the cross-repo graph.
"""
from config.settings import AgentRole


class ArchitectureSynthesizer:
    role = AgentRole.ARCHITECTURE_SYNTHESIZER

    def synthesize_module_doc(self, repo_id: str, graph_node_id: str) -> str:
        raise NotImplementedError(
            "Phase 5 — Architecture Synthesizer not yet implemented"
        )

    def synthesize_system_doc(self) -> str:
        raise NotImplementedError(
            "Phase 5 — Architecture Synthesizer not yet implemented"
        )
