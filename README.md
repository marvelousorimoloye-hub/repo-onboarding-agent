# Codebase Onboarding & Doc-Drift Agent Swarm

Status: **Phase 0 complete.** Everything under `agents/`, `ingestion/`
(except `repo_mapper.py`'s data model), and `orchestration/` is a skeleton
— method stubs that raise `NotImplementedError`, not working code yet.

## What's actually implemented (Phase 0)
- `config/settings.py` — model role mapping (Groq/Gemini), DB settings
- `db/connection.py`, `db/schema.sql` — Postgres + pgvector schema, `repo_id` tagged throughout
- `models/llm_clients.py` — Groq/Gemini client wrappers, dispatched by agent role
- `docs_templates/` — fixed doc structure (module doc + system-level doc)
- `ingestion/repo_mapper.py` — single-repo output data model only (`RepoGraph`/`RepoNode`/`RepoEdge`); no parsing logic yet

## Setup
```
cp .env.example .env   # fill in DB + API credentials
pip install -r requirements.txt
python main.py          # applies schema, verifies DB connectivity
```

## Next up
Phase 1 — implement `RepoMapper.build_graph()` (tree-sitter parsing for
Python/TypeScript), then `ingestion/cross_repo_linker.py`.
