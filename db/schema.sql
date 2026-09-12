-- Phase 0 schema. Multi-repo is baseline: every content table carries
-- repo_id from day one so retagging later isn't needed.
--
-- Note: the LangGraph Postgres checkpointer manages its own tables
-- (created automatically by langgraph-checkpoint-postgres's setup()) in
-- this same database — not defined here.

CREATE EXTENSION IF NOT EXISTS vector;

-- Repo registry (Phase 1)
CREATE TABLE IF NOT EXISTS repos (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            TEXT NOT NULL UNIQUE,
    git_url         TEXT NOT NULL,
    default_branch  TEXT NOT NULL DEFAULT 'main',
    status          TEXT NOT NULL DEFAULT 'active',  -- active | inactive (soft delete)
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_synced_at  TIMESTAMPTZ
);

-- Per-repo structural graph nodes (Phase 1: Repo Mapper output)
CREATE TABLE IF NOT EXISTS graph_nodes (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repo_id         UUID NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
    node_type       TEXT NOT NULL,   -- module | file | function | class | endpoint
    qualified_name  TEXT NOT NULL,   -- e.g. "payments.retry.RetryPolicy"
    file_path       TEXT NOT NULL,
    metadata        JSONB DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_graph_nodes_repo ON graph_nodes(repo_id);

-- Structural edges: within-repo (Phase 1) and cross-repo (Phase 1 extension)
CREATE TABLE IF NOT EXISTS graph_edges (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_node_id  UUID NOT NULL REFERENCES graph_nodes(id) ON DELETE CASCADE,
    target_node_id  UUID NOT NULL REFERENCES graph_nodes(id) ON DELETE CASCADE,
    edge_type       TEXT NOT NULL,   -- imports | calls | http_call | grpc_call | manual | defines
    is_cross_repo   BOOLEAN NOT NULL DEFAULT false,
    confidence      TEXT NOT NULL DEFAULT 'confirmed',  -- confirmed | inferred
    metadata        JSONB DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_graph_edges_source ON graph_edges(source_node_id);
CREATE INDEX IF NOT EXISTS idx_graph_edges_target ON graph_edges(target_node_id);

-- Semantic index (Phase 2)
CREATE TABLE IF NOT EXISTS code_chunks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repo_id         UUID NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
    graph_node_id   UUID REFERENCES graph_nodes(id) ON DELETE SET NULL,
    source_type     TEXT NOT NULL,   -- code | doc | pr_description | commit | api_spec
    file_path       TEXT,
    content         TEXT NOT NULL,
    embedding       vector(768),
    metadata        JSONB DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_code_chunks_repo ON code_chunks(repo_id);

-- Generated documentation (Phase 5)
CREATE TABLE IF NOT EXISTS doc_sections (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    repo_id         UUID REFERENCES repos(id) ON DELETE CASCADE,  -- NULL for system-level docs
    doc_type        TEXT NOT NULL,   -- module | system_level
    graph_node_id   UUID REFERENCES graph_nodes(id) ON DELETE SET NULL,
    title           TEXT NOT NULL,
    content_md      TEXT NOT NULL,
    last_verified_commit TEXT,
    last_verified_at     TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_doc_sections_repo ON doc_sections(repo_id);

-- Drift flags (Phase 7)
CREATE TABLE IF NOT EXISTS drift_flags (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_section_id  UUID NOT NULL REFERENCES doc_sections(id) ON DELETE CASCADE,
    triggering_repo_id UUID REFERENCES repos(id) ON DELETE SET NULL,
    triggering_commit  TEXT,
    flagged_text    TEXT NOT NULL,
    reason          TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'open',  -- open | resolved | dismissed
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_drift_flags_doc ON drift_flags(doc_section_id);

-- Access control (Phase 3). Deliberately minimal — this is the HOOK a real
-- auth/permissions system plugs into later, not a full IAM implementation.
-- Semantics: a user_id with ZERO rows here has unrestricted access to all
-- active repos (the sensible dev/test default, since no permission system
-- exists yet). A user_id with ANY rows is treated as an allowlist — only
-- the repos explicitly listed. This lets access control be introduced
-- without requiring every existing user/test setup to configure it first.
CREATE TABLE IF NOT EXISTS repo_access (
    user_id         TEXT NOT NULL,
    repo_id         UUID NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
    granted_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, repo_id)
);