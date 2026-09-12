"""
Cross-Repo Linker — Phase 1 extension.

Runs after repos are registered and individually mapped (RepoRegistry +
RepoMapper have already persisted each repo's file/function/class nodes).
This module adds ENDPOINT nodes (from OpenAPI specs / .proto files) and the
cross-repo edges that connect a call site in one repo to a definition in
another.

Three link strategies, in priority order:
1. Spec-based (confirmed) — parse OpenAPI/.proto definitions in each repo,
   match call-site path literals against them by normalized path.
2. String-literal fallback (inferred) — no spec match found; if a call
   site's literal contains another registered repo's name, link to that
   repo's synthetic "service root" node at lower confidence.
3. Manual override (confirmed) — links declared in a config file, for
   cases neither method catches.

Call-site literals themselves are NOT scanned here — RepoMapper (Phase 1)
already captures them during its single AST pass over each file
(file_node.metadata["call_site_literals"]), scoped to calls that look like
actual HTTP client calls. Re-scanning raw file text here would (a) double
the I/O/parse cost and (b) reintroduce the false-positive problem of
matching any quoted string that merely looks like a path.

Known limitations (deliberately out of scope for this pass):
- Path matching is literal/template-based, not a full OpenAPI path-matching
  spec implementation (no support for path-level parameter type
  constraints, wildcards beyond {param}/:param/<param> style).
- Dynamically constructed URLs (f-strings, template literals, variables)
  are never captured — see RepoMapper's docstring. They will not produce
  any edge, confirmed or inferred, rather than risk a wrong guess.
- Spec/proto file discovery still walks the full file list (via
  `git ls-files`, not a raw os.walk) on every run — true incremental
  discovery (skip repos/files unchanged since last link run) is deferred
  to Phase 7, which already owns the "track state between runs"
  infrastructure needed for incremental sync; building a separate cache
  here now would just be redone then.
"""
import os
import re
import json
import uuid
import subprocess
from dataclasses import dataclass

import yaml

from db.connection import connection
from ingestion.repo_registry import RepoRegistry, RepoRecord

OPENAPI_FILENAMES = {
    "openapi.yaml", "openapi.yml", "openapi.json",
    "swagger.yaml", "swagger.yml", "swagger.json",
}
# Real repos don't reliably use those exact filenames (e.g. Medusa ships
# docs/api/admin-spec3.yaml). As a broader net, any .yaml/.yml/.json file
# whose NAME hints at being a spec gets opened and checked for an actual
# `openapi`/`swagger` top-level key before being trusted — the filename
# hint is just a cheap prefilter so we're not parsing every JSON file in a
# large repo (package.json, tsconfig.json, locale files, etc.), not the
# final check.
OPENAPI_FILENAME_HINTS = ("spec", "openapi", "swagger")
OPENAPI_EXTENSIONS = (".yaml", ".yml", ".json")
PROTO_EXT = ".proto"
IGNORED_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"}

# Common path prefixes that vary between how a spec documents a route and
# how a caller actually reaches it (e.g. behind a gateway that adds /v1).
# Stripped before comparison — a fixed transformation, not fuzzy matching,
# so it doesn't add false-confirm risk the way looser matching would.
COMMON_PATH_PREFIXES = ("/v1", "/v2", "/v3", "/api")

MANUAL_OVERRIDES_PATH = os.getenv("CROSS_REPO_LINKS_CONFIG", "./cross_repo_links.yaml")


@dataclass
class CrossRepoEdge:
    id: str
    source_repo_id: str
    source_node_id: str
    target_repo_id: str
    target_node_id: str
    edge_type: str  # http_call | grpc_call | manual
    confidence: str  # confirmed | inferred


@dataclass
class _Endpoint:
    node_id: str
    repo_id: str
    method: str | None   # e.g. "get" — None for proto RPCs
    path: str            # normalized path, or "service.rpc" for proto


class CrossRepoLinker:
    def __init__(self):
        self._registry = RepoRegistry()

    def link(self, repo_ids: list[str] | None = None) -> list[CrossRepoEdge]:
        repos = self._registry.list_active_repos()
        if repo_ids is not None:
            repos = [r for r in repos if r.id in repo_ids]
        if len(repos) < 2:
            return []  # nothing to cross-link with fewer than 2 repos

        edges: list[CrossRepoEdge] = []

        # 1. Discover + persist endpoint nodes per repo (spec-based).
        endpoints_by_repo: dict[str, list[_Endpoint]] = {}
        for repo in repos:
            openapi_eps = self._discover_openapi_endpoints(repo)
            proto_eps = self._discover_proto_endpoints(repo)
            endpoints_by_repo[repo.id] = openapi_eps + proto_eps

        all_endpoints: list[_Endpoint] = [
            ep for eps in endpoints_by_repo.values() for ep in eps
        ]

        # Ensure every repo has a service-root node for the inferred-link
        # fallback and for manual overrides.
        service_root_by_repo: dict[str, str] = {
            repo.id: self._get_or_create_service_root_node(repo) for repo in repos
        }

        # 2. Match call sites (captured by RepoMapper's AST pass, read here
        #    from graph_nodes metadata) against spec (confirmed) or
        #    repo-name string match (inferred).
        for repo in repos:
            call_sites = self._load_call_sites(repo.id)
            for source_node_id, literal in call_sites:
                match = self._match_against_specs(literal, all_endpoints, exclude_repo_id=repo.id)
                if match is not None:
                    edges.append(self._build_edge(
                        source_repo_id=repo.id, source_node_id=source_node_id,
                        target_repo_id=match.repo_id, target_node_id=match.node_id,
                        edge_type="http_call" if match.method else "grpc_call",
                        confidence="confirmed",
                    ))
                    continue

                inferred_target_repo = self._match_repo_name_in_literal(
                    literal, [r for r in repos if r.id != repo.id]
                )
                if inferred_target_repo is not None:
                    edges.append(self._build_edge(
                        source_repo_id=repo.id, source_node_id=source_node_id,
                        target_repo_id=inferred_target_repo.id,
                        target_node_id=service_root_by_repo[inferred_target_repo.id],
                        edge_type="http_call",
                        confidence="inferred",
                    ))

        # 3. Manual overrides.
        edges.extend(self._apply_manual_overrides(repos, service_root_by_repo))

        for edge in edges:
            self._persist_edge(edge)
        return edges

    # --- spec discovery ------------------------------------------------

    def _discover_openapi_endpoints(self, repo: RepoRecord) -> list[_Endpoint]:
        endpoints: list[_Endpoint] = []
        for abs_path in self._walk_files(repo.local_path):
            basename = os.path.basename(abs_path)
            basename_lower = basename.lower()
            ext = os.path.splitext(basename_lower)[1]

            is_exact_match = basename_lower in OPENAPI_FILENAMES
            is_hinted_candidate = (
                ext in OPENAPI_EXTENSIONS
                and any(hint in basename_lower for hint in OPENAPI_FILENAME_HINTS)
            )
            if not (is_exact_match or is_hinted_candidate):
                continue

            try:
                with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                    if ext == ".json":
                        spec = json.load(f)
                    else:
                        spec = yaml.safe_load(f)
            except Exception:
                continue  # malformed/non-YAML-or-JSON file — skip rather than fail the whole run

            if not isinstance(spec, dict):
                continue  # empty/malformed file — not a spec regardless of filename

            if is_hinted_candidate and not is_exact_match and not ("openapi" in spec or "swagger" in spec):
                # Hinted candidate turned out not to actually be a spec
                # (e.g. some unrelated "test-spec-runner.json") — exact
                # filename matches skip this check since those names are
                # unambiguous by convention.
                continue

            for path, path_item in (spec.get("paths") or {}).items():
                if not isinstance(path_item, dict):
                    continue
                for method in path_item.keys():
                    if method.lower() not in {"get", "post", "put", "delete", "patch"}:
                        continue
                    node_id = self._persist_endpoint_node(
                        repo.id, method=method.lower(), path=self._normalize_path(path),
                        qualified_name=f"{repo.name}:{method.upper()} {path}",
                    )
                    endpoints.append(_Endpoint(
                        node_id=node_id, repo_id=repo.id,
                        method=method.lower(), path=self._normalize_path(path),
                    ))
        return endpoints

    def _discover_proto_endpoints(self, repo: RepoRecord) -> list[_Endpoint]:
        endpoints: list[_Endpoint] = []
        service_re = re.compile(r"service\s+(\w+)\s*\{(.*?)\}", re.DOTALL)
        rpc_re = re.compile(r"rpc\s+(\w+)\s*\(")

        for abs_path in self._walk_files(repo.local_path):
            if not abs_path.endswith(PROTO_EXT):
                continue
            with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            for service_match in service_re.finditer(content):
                service_name, body = service_match.group(1), service_match.group(2)
                for rpc_match in rpc_re.finditer(body):
                    rpc_name = rpc_match.group(1)
                    qualified = f"{service_name}.{rpc_name}"
                    node_id = self._persist_endpoint_node(
                        repo.id, method=None, path=qualified,
                        qualified_name=f"{repo.name}:{qualified}",
                    )
                    endpoints.append(_Endpoint(
                        node_id=node_id, repo_id=repo.id, method=None, path=qualified,
                    ))
        return endpoints

    # --- call site loading (from RepoMapper's AST pass) --------------------

    def _load_call_sites(self, repo_id: str) -> list[tuple[str, str]]:
        """Returns (source_node_id, literal) pairs from graph_nodes.metadata,
        as captured by RepoMapper during its AST pass. No file I/O here."""
        results = []
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, metadata FROM graph_nodes
                    WHERE repo_id = %s AND node_type = 'file'
                    """,
                    (repo_id,),
                )
                rows = cur.fetchall()
        for node_id, metadata in rows:
            literals = (metadata or {}).get("call_site_literals", [])
            for literal in literals:
                results.append((str(node_id), literal))
        return results

    # --- matching --------------------------------------------------------

    def _match_against_specs(
        self, literal: str, endpoints: list[_Endpoint], exclude_repo_id: str
    ) -> _Endpoint | None:
        normalized = self._normalize_path(self._strip_host(literal))
        for ep in endpoints:
            if ep.repo_id == exclude_repo_id:
                continue
            if ep.path == normalized:
                return ep
        return None

    def _match_repo_name_in_literal(
        self, literal: str, candidate_repos: list[RepoRecord]
    ) -> RepoRecord | None:
        literal_norm = literal.lower().replace("-", "").replace("_", "")
        for repo in candidate_repos:
            name_norm = repo.name.lower().replace("-", "").replace("_", "")
            if name_norm and name_norm in literal_norm:
                return repo
        return None

    @staticmethod
    def _normalize_path(path: str) -> str:
        # Strip common version/gateway prefixes first — a fixed
        # transformation, applied before segment normalization, so
        # "/v1/orders/{id}" (spec) and "/orders/42" (caller, behind a
        # gateway that strips /v1) still compare equal.
        for prefix in COMMON_PATH_PREFIXES:
            if path.startswith(prefix + "/") or path == prefix:
                path = path[len(prefix):]
                break

        # Collapse {param} / :param / <param> style segments to '*' so
        # "/orders/{id}" and "/orders/:id" and "/orders/42" all compare equal.
        segments = path.strip("/").split("/")
        normalized = [
            "*" if (s.startswith("{") or s.startswith(":") or s.startswith("<") or s.isdigit())
            else s
            for s in segments
        ]
        return "/" + "/".join(normalized)

    @staticmethod
    def _strip_host(literal: str) -> str:
        if literal.startswith("http://") or literal.startswith("https://"):
            without_scheme = literal.split("://", 1)[1]
            path_part = without_scheme.split("/", 1)
            return "/" + path_part[1] if len(path_part) > 1 else "/"
        return literal

    # --- manual overrides --------------------------------------------------

    def _apply_manual_overrides(
        self, repos: list[RepoRecord], service_root_by_repo: dict[str, str]
    ) -> list[CrossRepoEdge]:
        if not os.path.exists(MANUAL_OVERRIDES_PATH):
            return []

        with open(MANUAL_OVERRIDES_PATH, "r") as f:
            config = yaml.safe_load(f) or {}

        by_name = {r.name: r for r in repos}
        edges = []
        for link in config.get("links", []):
            source_repo = by_name.get(link.get("source_repo"))
            target_repo = by_name.get(link.get("target_repo"))
            if source_repo is None or target_repo is None:
                continue  # unknown repo name in config — skip rather than fail the run
            edges.append(self._build_edge(
                source_repo_id=source_repo.id,
                source_node_id=service_root_by_repo[source_repo.id],
                target_repo_id=target_repo.id,
                target_node_id=service_root_by_repo[target_repo.id],
                edge_type=link.get("edge_type", "manual"),
                confidence="confirmed",
            ))
        return edges

    # --- persistence helpers ------------------------------------------------

    def _walk_files(self, root: str):
        """Prefers `git ls-files` (fast — no per-directory stat storm on
        large repos, and respects .gitignore automatically), falls back to
        os.walk for non-git directories."""
        try:
            result = subprocess.run(
                ["git", "-C", root, "ls-files"],
                check=True, capture_output=True, text=True,
            )
            for rel_path in result.stdout.splitlines():
                if rel_path:
                    yield os.path.join(root, rel_path)
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        for dirpath, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
            for fname in files:
                yield os.path.join(dirpath, fname)

    def _persist_endpoint_node(
        self, repo_id: str, method: str | None, path: str, qualified_name: str
    ) -> str:
        node_id = str(uuid.uuid4())
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO graph_nodes (id, repo_id, node_type, qualified_name, file_path, metadata)
                    VALUES (%s, %s, 'endpoint', %s, %s, %s)
                    """,
                    (node_id, repo_id, qualified_name, "", json.dumps({"method": method, "path": path})),
                )
        return node_id

    def _get_or_create_service_root_node(self, repo: RepoRecord) -> str:
        qualified_name = f"{repo.name}::__service_root__"
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id FROM graph_nodes
                    WHERE repo_id = %s AND qualified_name = %s
                    """,
                    (repo.id, qualified_name),
                )
                row = cur.fetchone()
                if row:
                    return str(row[0])

                node_id = str(uuid.uuid4())
                cur.execute(
                    """
                    INSERT INTO graph_nodes (id, repo_id, node_type, qualified_name, file_path, metadata)
                    VALUES (%s, %s, 'module', %s, '', '{}'::jsonb)
                    """,
                    (node_id, repo.id, qualified_name),
                )
        return node_id

    def _build_edge(
        self, source_repo_id, source_node_id, target_repo_id, target_node_id,
        edge_type, confidence,
    ) -> CrossRepoEdge:
        return CrossRepoEdge(
            id=str(uuid.uuid4()),
            source_repo_id=source_repo_id,
            source_node_id=source_node_id,
            target_repo_id=target_repo_id,
            target_node_id=target_node_id,
            edge_type=edge_type,
            confidence=confidence,
        )

    def _persist_edge(self, edge: CrossRepoEdge) -> None:
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO graph_edges
                        (id, source_node_id, target_node_id, edge_type, is_cross_repo, confidence)
                    VALUES (%s, %s, %s, %s, true, %s)
                    """,
                    (edge.id, edge.source_node_id, edge.target_node_id, edge.edge_type, edge.confidence),
                )