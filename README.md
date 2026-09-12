# Codebase Onboarding & Doc-Drift Agent Swarm

A LangGraph-based multi-agent system that maps codebases — including
cross-repo dependencies — semantically indexes them, and answers
engineering questions with grounded, cited answers. Built for engineering
onboarding and documentation-drift detection at multi-repo orgs.

Multi-repo support is baseline architecture from day one, not a v2
add-on — a single-repo tool isn't a compelling pitch for real engineering
orgs, most of which run several interconnected services.

## Status

Every phase below has been tested against real forked repositories (not
just toy examples), which is deliberate: nearly every bug listed under
"real bugs found and fixed" only surfaced because of that discipline —
none of them would have been caught by code review alone.

### ✅ Phase 0 — Foundations
- Config: per-agent-role model dispatch (Groq for fast/high-frequency
  roles, Gemini for careful-reasoning roles), swappable to stronger
  models later via a one-line config change
- Postgres + pgvector connection layer
- Schema: `repos`, `graph_nodes`, `graph_edges`, `code_chunks`,
  `doc_sections`, `drift_flags`, `repo_access`

### ✅ Phase 1 — Repo Mapper, Repo Registry, Cross-Repo Linker
- Tree-sitter parsing (Python, TypeScript/JavaScript): files, functions,
  classes, imports (relative/absolute/bare styles), same-file calls
- HTTP call-site detection by **argument shape**, not hardcoded client
  library names — resolves plain strings, f-strings/template literals,
  string concatenation, and `os.getenv`/`process.env` defaults, with
  branch-aware variable resolution and a general resolution-cycle guard
- Cross-repo linking, three tiers: **confirmed** (OpenAPI/Swagger spec
  match, including content-sniffed non-standard filenames), **inferred**
  (repo-name match), **manual override** (YAML config)
- Repo registry: register/sync/deregister/purge, `git ls-files`-based
  discovery, Windows-safe cleanup (read-only `.git/objects` handling)

**Real bugs found and fixed:** absolute/bare Python import styles not
resolving; HTTP-client detection missing unlisted libraries (superagent,
etc.) — replaced with the shape-based approach above; a self-referential
assignment (`url = url + "/x"`) causing infinite recursion — fixed with
an ancestor-exclusion check, then hardened further with a general cycle
guard when a second, different recursion pattern appeared; Windows path
separators (`\` vs `/`) silently breaking cross-module path comparisons.

### ✅ Phase 2 — Semantic Indexer
- Function/class-precise chunking (via Phase 1's line-range metadata),
  falling back to fixed-size windowed chunking where precision isn't
  available
- Doc chunking (markdown, heading-based), commit message chunking, PR
  description chunking (GitHub REST API), OpenAPI/Swagger spec chunking
  (one embeddable chunk per endpoint)
- pgvector similarity search, single-repo and cross-repo scoped
- Self-throttling + retry-with-backoff against embedding API rate limits

**Real bugs found and fixed:** stale embedding model name
(`text-embedding-004` → `gemini-embedding-001`); embedding dimensionality
mismatch (model's native 3072-dim vs. schema's 768-dim, fixed via
`output_dimensionality`); two separate pgvector/psycopg2 type-cast
gotchas (`vector` and `uuid` comparisons both need explicit `::type`
casts psycopg2 won't infer on their own); precise chunking silently
degrading to windowed-only on Windows due to the path-separator bug
above.

### ✅ Phase 3 — Router, Retrieval Agent, Answer Synthesizer
- Router: classifies question type, disambiguates repo scope (skips the
  LLM call entirely when only one repo is registered; defaults to
  broad search + a `needs_clarification` flag when genuinely ambiguous,
  rather than blocking)
- Retrieval Agent: repo-scoped search with an access-control hook
  (`repo_access` table — a zero-row user is unrestricted by default;
  any row switches that user to an allowlist)
- Answer Synthesizer: cited answers resolving to real `file:line`
  locations (or commit sha / PR number for non-code sources), not
  opaque reference numbers

**Validated:** both Router branches (ambiguous and confident scoping),
both Retrieval Agent branches (unrestricted default and allowlist
filtering), citation resolution.

### 🔶 Phase 4 — Trace Agent (mostly validated, one open item)
- Multi-hop investigation across `graph_edges`. A function/class node's
  own direct edges are often incomplete (e.g. `CALLS` edges only cover
  same-file bare calls, not attribute/method calls or cross-file calls),
  so the agent folds in its **enclosing file's** edges as additional,
  clearly-labeled lower-precision hop candidates
- Stops on: sufficient context found, dead end, cycle, max hops reached,
  or a safe bail-out on an unparseable/invalid model decision

**Validated:** a real 3-hop trace (function → file → class) correctly
landing on precise implementation details, verified word-for-word
against real code (circuit-breaker failure threshold and cooldown
duration); correct refusal to hallucinate when `max_hops` was
deliberately set too low to reach the answer.

**Open investigation:** a `@dataclass`-decorated class isn't showing up
as a graph node. Root cause identified — tree-sitter-python wraps a
decorated definition in an extra `decorated_definition` AST layer, which
silently defeats the existing depth-based top-level check — and a fix
has been written (`_effective_depth`), but a re-registration with the
fix in place still showed unchanged node counts. Not yet resolved
whether the fix isn't taking effect (file not actually replaced, stale
bytecode) or the diagnosis needs revisiting. **Next step when picking
this back up:** run `debug_trace_edges.py PetStoreAPI data.py` to look
directly at what's actually in the graph before changing anything else.

## Known, deliberate boundaries (not oversights)

- **Cross-file constant resolution is explicitly out of `RepoMapper`'s
  scope.** A URL host imported from another file won't resolve
  statically — this was a deliberate call to defer that reasoning to the
  Trace Agent (which has retrieval to do it properly) rather than build
  a narrower, static reimplementation into Phase 1.
- **TS/JS decorator handling** analogous to the Python
  `_effective_depth` fix above is unverified — flagged as a risk to
  watch for, not assumed already fixed.
- Only Python and TypeScript/JavaScript get structural (function/class)
  parsing. Doc/commit/PR-description/API-spec chunking work regardless
  of the surrounding code's language.
- No true incremental sync yet — `RepoRegistry.sync()` always does a
  full re-map. Deferred to Phase 7, where the Drift Detector's
  PR-triggered use case actually needs the speed.

## Repository layout

```
config/          settings, model role mapping
db/              Postgres connection, schema.sql
models/          LLM client dispatch (Groq/Gemini)
ingestion/       RepoMapper, RepoRegistry, CrossRepoLinker, SemanticIndexer
agents/          Router, RetrievalAgent, AnswerSynthesizer, TraceAgent
orchestration/   LangGraph wiring (Phase 8 — not yet built)
docs_templates/  fixed doc structure for generated docs (Phase 5)
test_*.py        phase-by-phase smoke tests, run directly against real repos
```

## Setup

1. `cp .env.example .env` and fill in Postgres credentials, `GROQ_API_KEY`,
   `GEMINI_API_KEY`, and optionally `GITHUB_TOKEN` (raises the PR-fetch
   rate limit from 60/hr to 5000/hr and is required for private repos)
2. `pip install -r requirements.txt --break-system-packages`
3. `python main.py` — applies the schema, confirms DB connectivity
4. Register a repo and run the phase-appropriate test script (see
   `test_phase1.py` through `test_phase4.py` for examples)

## Next steps

- **Resolve the Phase 4 decorator investigation** before calling Phase 4
  fully closed
- **Phase 5 — Architecture Synthesizer:** auto-generate per-module docs
  and a system-level architecture doc from the dependency graph
- **Phase 6 — Verifier/Grounding Agent:** check generated doc claims
  against actual code before anything is marked published
- **Phase 7 — Drift Detector:** flag doc/code divergence on PR/merge,
  with cross-repo fan-out (a change in one repo invalidating docs in
  another)
- **Phase 8 — Orchestration hardening:** wire everything into LangGraph
  state graphs with Postgres-backed checkpointing for resumable traces
- **Phase 9 — Sellable features:** PR-triggered drift-comment bot,
  confidence scoring, doc staleness dashboard, ADR ingestion, Slack
  integration, onboarding-path generation, IDE plugin, human-in-the-loop
  approval queue