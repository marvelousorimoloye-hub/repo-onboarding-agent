"""
Semantic Indexer — Phase 2.

Chunks code, existing docs, and commit history for a repo, embeds each
chunk (Gemini's text-embedding-004 — Groq doesn't offer an embeddings
endpoint), and persists to the `code_chunks` table (pgvector), tagged with
repo_id so retrieval can be scoped correctly per repo or across repos.

Scope of this pass:
- Code chunking is FUNCTION/CLASS-BOUNDARY-PRECISE where RepoMapper (Phase
  1) captured line ranges for a node (function_definition,
  class_definition, etc. — see RepoMapper._line_range_metadata). Each such
  chunk is tagged with graph_node_id, linking it back to the exact graph
  node it came from. For a file with no captured functions/classes at all
  (a pure script, or a language/shape RepoMapper doesn't parse function
  boundaries for), falls back to fixed-size line-window chunking so
  coverage isn't lost — see _chunk_source_files_windowed.
- Doc chunking is heading-based for markdown, not line-window — headings
  are a much more natural semantic boundary for docs than for code.
- Commit messages are indexed (source_type='commit') since they're free
  to read from the local git checkout. PR descriptions are ALSO indexed
  (source_type='pr_description'), via the GitHub REST API — but only for
  github.com-hosted repos; see _chunk_pr_descriptions for the graceful
  degradation when a repo isn't on GitHub or the API call fails.
- File discovery mirrors RepoMapper's `git ls-files` preference (fast,
  respects .gitignore), independently implemented here rather than
  imported, since this module walks a different extension set (code +
  docs, not just RepoMapper's parseable languages) and doesn't need
  RepoMapper's tree-sitter machinery at all — the two modules are natural
  siblings that both read repo trees but need different content.
"""
import os
import re
import json
import time
import subprocess
import uuid

import requests
import yaml
from google import genai
from google.genai import types

from config.settings import KEYS, EMBEDDING_MODEL_NAME, EMBEDDING_DIM, GITHUB_TOKEN
from db.connection import connection
from ingestion.repo_registry import RepoRegistry

CODE_EXTENSIONS = {".py", ".ts", ".tsx", ".js", ".jsx"}
DOC_EXTENSIONS = {".md", ".mdx", ".rst"}

# Same filename/content-sniffing definition CrossRepoLinker (Phase 1) uses
# for spec discovery — deliberately redefined here rather than imported
# from ingestion.cross_repo_linker. This module already independently
# walks files instead of importing RepoMapper's walker (see module
# docstring: the two ingestion modules are treated as decoupled siblings
# that both read repo trees but need different content) — same reasoning
# applies here. Duplication of these few constants is a small, known
# trade-off for that decoupling.
OPENAPI_FILENAMES = {
    "openapi.yaml", "openapi.yml", "openapi.json",
    "swagger.yaml", "swagger.yml", "swagger.json",
}
OPENAPI_FILENAME_HINTS = ("spec", "openapi", "swagger")
OPENAPI_EXTENSIONS = (".yaml", ".yml", ".json")
IGNORED_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"}

CHUNK_LINES = 60
CHUNK_OVERLAP_LINES = 10
MAX_COMMITS = 200  # avoid indexing a huge repo's entire history by default
MAX_PRS = 100  # same reasoning, for PR description ingestion

# A precise function/class chunk larger than this still gets sub-chunked
# via the same windowing logic — an oversized function is otherwise a
# single low-precision embedding covering way too much ground.
MAX_PRECISE_CHUNK_LINES = 120

# The observed free-tier quota is 100 embed_content REQUESTS per minute —
# not per text. Since the quota is per-request, a bigger batch directly
# reduces how many requests a given amount of content needs, which is the
# most direct lever against hitting it. 100 matches the API's own
# documented per-request batch cap.
EMBED_BATCH_SIZE = 100

# Self-imposed pacing: stay a bit under the observed 100/min quota rather
# than firing every batch back-to-back and relying entirely on
# retry-after-429 to recover. RATE_LIMIT_WINDOW_SECONDS/EMBED_REQUESTS_PER_MINUTE
# together define a sliding-window throttle in _throttle_for_rate_limit.
RATE_LIMIT_WINDOW_SECONDS = 60
EMBED_REQUESTS_PER_MINUTE = 90  # margin under the observed 100/min limit

# Retry-with-backoff as a safety net for whatever the pacing above doesn't
# catch (quota shared across other concurrent usage, a burst right at the
# window boundary, etc.) — not the primary defense, self-throttling is.
MAX_RATE_LIMIT_RETRIES = 5
RATE_LIMIT_BASE_BACKOFF_SECONDS = 10


class SemanticIndexer:
    def __init__(self):
        self._client = genai.Client(api_key=KEYS.gemini_api_key)
        self._registry = RepoRegistry()
        self._request_timestamps: list[float] = []  # sliding window for self-throttling

    def index_repo(self, repo_id: str) -> None:
        """Ordering dependency worth being explicit about: this reads
        graph_nodes (for precise function/class line ranges) via
        _fetch_function_class_nodes, so the repo must already have been
        mapped — RepoRegistry.register()/sync() (Phase 1) needs to have
        run for this repo_id first. If it hasn't, code chunking still
        works correctly (just falls back to windowed chunking for every
        file, same as any file RepoMapper found no functions in)."""
        record = self._registry.get_repo(repo_id)
        if record is None:
            raise ValueError(f"No repo registered with id={repo_id}")

        self._clear_existing_chunks(repo_id)

        chunks: list[dict] = []
        chunks.extend(self._chunk_source_files(repo_id, record.local_path))
        chunks.extend(self._chunk_doc_files(record.local_path))
        chunks.extend(self._chunk_api_specs(record.local_path))
        chunks.extend(self._chunk_commit_messages(record.local_path))
        chunks.extend(self._chunk_pr_descriptions(record.git_url))

        self._embed_and_persist(repo_id, chunks)

    def get_chunk_counts(self, repo_id: str) -> dict[str, int]:
        """Counts persisted chunks by source_type. Exists because
        graceful degradation (a design choice throughout this module —
        a failed fetch returns [] rather than raising) has a real cost:
        it makes 'this source produced nothing' indistinguishable from
        'this source genuinely has nothing' when you're only looking at
        search results. This gives a direct way to check, rather than
        inferring PR/spec/commit coverage from whether they happened to
        rank in a particular query's top results."""
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT source_type, count(*) FROM code_chunks WHERE repo_id = %s GROUP BY source_type",
                    (repo_id,),
                )
                rows = cur.fetchall()
        return {source_type: count for source_type, count in rows}

    def search(self, query: str, repo_ids: list[str] | None = None, top_k: int = 10) -> list[dict]:
        query_embedding = self._embed_texts([query])[0]
        with connection() as conn:
            with conn.cursor() as cur:
                if repo_ids:
                    cur.execute(
                        """
                        SELECT id, repo_id, graph_node_id, source_type, file_path, content, metadata,
                               1 - (embedding <=> %s::vector) AS similarity
                        FROM code_chunks
                        WHERE repo_id = ANY(%s::uuid[])
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (query_embedding, repo_ids, query_embedding, top_k),
                    )
                else:
                    cur.execute(
                        """
                        SELECT id, repo_id, graph_node_id, source_type, file_path, content, metadata,
                               1 - (embedding <=> %s::vector) AS similarity
                        FROM code_chunks
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (query_embedding, query_embedding, top_k),
                    )
                rows = cur.fetchall()
        return [
            {
                "id": str(r[0]), "repo_id": str(r[1]),
                "graph_node_id": str(r[2]) if r[2] else None,
                "source_type": r[3], "file_path": r[4], "content": r[5],
                "metadata": r[6] or {}, "similarity": float(r[7]),
            }
            for r in rows
        ]

    # --- chunking: code --------------------------------------------------

    def _chunk_source_files(self, repo_id: str, local_path: str) -> list[dict]:
        """One chunk per function/class where RepoMapper captured a line
        range for it (graph_node_id-linked, so a search hit can be traced
        straight back to the exact graph node). Files with no such nodes
        at all fall back to fixed-size windowed chunking, so coverage
        never silently drops to zero for a file RepoMapper's parser
        didn't extract functions from."""
        nodes_by_file = self._fetch_function_class_nodes(repo_id)
        chunks = []

        for rel_path in self._walk_files(local_path):
            ext = os.path.splitext(rel_path)[1]
            if ext not in CODE_EXTENSIONS:
                continue
            abs_path = os.path.join(local_path, rel_path)
            try:
                with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
            except (OSError, UnicodeDecodeError):
                continue  # unreadable file — skip, don't abort the whole index
            if not lines:
                continue

            file_nodes = nodes_by_file.get(rel_path, [])
            if file_nodes:
                chunks.extend(self._chunk_by_node_ranges(rel_path, lines, file_nodes))
            else:
                chunks.extend(self._chunk_lines_windowed(rel_path, lines, source_type="code"))

        return chunks

    def _fetch_function_class_nodes(self, repo_id: str) -> dict[str, list[dict]]:
        """Returns {file_path: [{graph_node_id, qualified_name, start_line,
        end_line}, ...]} for every FUNCTION/CLASS node RepoMapper captured
        a line range for. Nodes without a usable range (metadata missing
        start_line/end_line — shouldn't happen for anything created after
        the Phase 1 line-range addition, but old data or an edge case
        could lack it) are skipped rather than guessed at."""
        result: dict[str, list[dict]] = {}
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, file_path, qualified_name, metadata
                    FROM graph_nodes
                    WHERE repo_id = %s AND node_type IN ('function', 'class')
                    """,
                    (repo_id,),
                )
                rows = cur.fetchall()
        for node_id, file_path, qualified_name, metadata in rows:
            metadata = metadata or {}
            start_line = metadata.get("start_line")
            end_line = metadata.get("end_line")
            if start_line is None or end_line is None:
                continue
            result.setdefault(file_path, []).append({
                "graph_node_id": str(node_id),
                "qualified_name": qualified_name,
                "start_line": start_line,
                "end_line": end_line,
            })
        return result

    def _chunk_by_node_ranges(self, rel_path: str, lines: list[str], file_nodes: list[dict]) -> list[dict]:
        chunks = []
        for node in file_nodes:
            start, end = node["start_line"], node["end_line"]
            span = lines[max(start - 1, 0):end]
            content = "".join(span).strip()
            if not content:
                continue
            if end - start + 1 > MAX_PRECISE_CHUNK_LINES:
                # Oversized function/class — sub-chunk it rather than
                # embedding one huge low-precision blob, but keep each
                # piece linked back to the same graph node.
                for sub in self._chunk_lines_windowed(rel_path, span, source_type="code"):
                    sub["graph_node_id"] = node["graph_node_id"]
                    sub["metadata"]["qualified_name"] = node["qualified_name"]
                    chunks.append(sub)
            else:
                chunks.append({
                    "source_type": "code",
                    "file_path": rel_path,
                    "content": content,
                    "graph_node_id": node["graph_node_id"],
                    "metadata": {
                        "qualified_name": node["qualified_name"],
                        "start_line": start,
                        "end_line": end,
                    },
                })
        return chunks

    def _chunk_lines_windowed(self, rel_path: str, lines: list[str], source_type: str) -> list[dict]:
        """Fixed-size fallback chunking — used for files RepoMapper found
        no functions/classes in, and for sub-chunking an oversized
        function/class."""
        chunks = []
        step = max(CHUNK_LINES - CHUNK_OVERLAP_LINES, 1)
        for start in range(0, len(lines), step):
            window = lines[start:start + CHUNK_LINES]
            content = "".join(window).strip()
            if content:
                chunks.append({
                    "source_type": source_type,
                    "file_path": rel_path,
                    "content": content,
                    "graph_node_id": None,
                    "metadata": {"start_line": start + 1, "end_line": start + len(window)},
                })
            if start + CHUNK_LINES >= len(lines):
                break
        return chunks

    # --- chunking: docs ----------------------------------------------------

    def _chunk_doc_files(self, local_path: str) -> list[dict]:
        chunks = []
        for rel_path in self._walk_files(local_path):
            ext = os.path.splitext(rel_path)[1]
            if ext not in DOC_EXTENSIONS:
                continue
            abs_path = os.path.join(local_path, rel_path)
            try:
                with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except (OSError, UnicodeDecodeError):
                continue

            for section in self._split_by_headings(content):
                section = section.strip()
                if section:
                    chunks.append({
                        "source_type": "doc",
                        "file_path": rel_path,
                        "content": section,
                        "graph_node_id": None,
                        "metadata": {},
                    })
        return chunks

    @staticmethod
    def _split_by_headings(content: str) -> list[str]:
        """Splits markdown on lines starting with '#' — a section per
        heading. Falls back to fixed-size chunks for any section that's
        still very long (avoids one giant unstructured doc becoming a
        single oversized, low-precision chunk)."""
        lines = content.splitlines()
        sections: list[list[str]] = [[]]
        for line in lines:
            if line.startswith("#") and sections[-1]:
                sections.append([])
            sections[-1].append(line)

        result = []
        for section_lines in sections:
            text = "\n".join(section_lines)
            if len(section_lines) <= CHUNK_LINES:
                result.append(text)
            else:
                step = max(CHUNK_LINES - CHUNK_OVERLAP_LINES, 1)
                for start in range(0, len(section_lines), step):
                    result.append("\n".join(section_lines[start:start + CHUNK_LINES]))
                    if start + CHUNK_LINES >= len(section_lines):
                        break
        return result

    # --- chunking: API specs ------------------------------------------

    def _chunk_api_specs(self, local_path: str) -> list[dict]:
        """Indexes OpenAPI/Swagger spec files as searchable content — one
        chunk per endpoint (method + path + summary/description), not the
        raw YAML dumped as a single blob. Closes a real gap found via
        testing: a query like "how do I get a pet by id" has nothing to
        match if the file that actually DEFINES that endpoint is never
        embedded — which was the case before this method existed, even
        though CrossRepoLinker (Phase 1) already reads the same files for
        structural endpoint discovery. Same graceful-degradation pattern
        as the rest of this module: a malformed or non-spec file that
        happens to match the filename hint is skipped, not fatal."""
        chunks = []
        for rel_path in self._walk_files(local_path):
            basename_lower = os.path.basename(rel_path).lower()
            ext = os.path.splitext(basename_lower)[1]
            is_exact = basename_lower in OPENAPI_FILENAMES
            is_hinted = ext in OPENAPI_EXTENSIONS and any(
                hint in basename_lower for hint in OPENAPI_FILENAME_HINTS
            )
            if not (is_exact or is_hinted):
                continue

            abs_path = os.path.join(local_path, rel_path)
            try:
                with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                    spec = json.load(f) if ext == ".json" else yaml.safe_load(f)
            except Exception:
                continue
            if not isinstance(spec, dict):
                continue
            if is_hinted and not is_exact and not ("openapi" in spec or "swagger" in spec):
                continue  # hinted filename that isn't actually a spec — skip

            for path, path_item in (spec.get("paths") or {}).items():
                if not isinstance(path_item, dict):
                    continue
                for method, operation in path_item.items():
                    if method.lower() not in {"get", "post", "put", "delete", "patch"}:
                        continue
                    if not isinstance(operation, dict):
                        continue
                    summary = operation.get("summary", "")
                    description = operation.get("description", "")
                    content = f"{method.upper()} {path}: {summary}\n{description}".strip()
                    if content:
                        chunks.append({
                            "source_type": "api_spec",
                            "file_path": rel_path,
                            "content": content,
                            "graph_node_id": None,
                            "metadata": {"method": method.lower(), "path": path},
                        })
        return chunks

    # --- chunking: commit history --------------------------------------

    def _chunk_commit_messages(self, local_path: str) -> list[dict]:
        try:
            result = subprocess.run(
                ["git", "-C", local_path, "log", f"-{MAX_COMMITS}",
                 "--pretty=format:%H%x1f%s%x1f%b%x1e"],
                capture_output=True, text=True, check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return []  # not a git repo, or git unavailable — skip, not fatal

        chunks = []
        for entry in result.stdout.split("\x1e"):
            entry = entry.strip()
            if not entry:
                continue
            parts = entry.split("\x1f")
            if len(parts) < 2:
                continue
            sha, subject = parts[0], parts[1]
            body = parts[2] if len(parts) > 2 else ""
            content = f"{subject}\n{body}".strip()
            if not content:
                continue
            chunks.append({
                "source_type": "commit",
                "file_path": None,
                "content": content,
                "graph_node_id": None,
                "metadata": {"commit_sha": sha},
            })
        return chunks

    # --- chunking: PR descriptions ----------------------------------------

    def _chunk_pr_descriptions(self, git_url: str) -> list[dict]:
        """Fetches PR titles+bodies via the GitHub REST API — the "why was
        this changed" signal that neither commit messages (often terse)
        nor the code itself (shows what, not why) reliably capture. This
        was part of Phase 2's original scope from the very first plan
        ("chunk code + docs + PR descriptions + commit messages"),
        deferred earlier in this build only because it needed its own
        integration surface (auth, a different API entirely) — now that
        the rest of Phase 2 is validated, completing it rather than
        leaving it open.

        Only works for github.com-hosted repos — anything else (a local
        fixture path used for testing, GitLab, Bitbucket) is skipped
        silently via _parse_github_owner_repo returning None. That's a
        genuine "nothing to fetch here," not an error; a repo not being
        on GitHub doesn't mean indexing should fail.

        Unauthenticated requests are capped at 60/hour by GitHub — fine
        for occasional use, but every failure mode here (rate limit hit,
        network issue, repo not found, malformed response) degrades to
        "return whatever was fetched so far" rather than raising, same
        principle as everywhere else in this module: partial coverage,
        never a hard failure that blocks the rest of indexing."""
        owner_repo = self._parse_github_owner_repo(git_url)
        if owner_repo is None:
            print(f"[pr_descriptions] {git_url} doesn't look like a github.com "
                  f"URL — skipping (not a bug, just not fetchable here).")
            return []
        owner, repo = owner_repo

        headers = {"Accept": "application/vnd.github+json"}
        if GITHUB_TOKEN:
            headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

        chunks = []
        page = 1
        fetched = 0
        while fetched < MAX_PRS:
            try:
                response = requests.get(
                    f"https://api.github.com/repos/{owner}/{repo}/pulls",
                    params={"state": "all", "per_page": 100, "page": page},
                    headers=headers,
                    timeout=10,
                )
            except requests.RequestException as e:
                print(f"[pr_descriptions] {owner}/{repo}: network error ({e}) — "
                      f"stopping with {fetched} PRs fetched so far.")
                break

            if response.status_code != 200:
                # Rate-limited, repo not found/private without a token,
                # etc. — same principle: stop here, don't fail the run.
                # Printed rather than silent, so "API call actually
                # failed" is distinguishable from "this repo genuinely
                # has zero PRs" (see the page==1-and-empty case below).
                print(f"[pr_descriptions] {owner}/{repo}: GitHub API returned "
                      f"{response.status_code} — {response.text[:200]}")
                break

            try:
                prs = response.json()
            except ValueError:
                print(f"[pr_descriptions] {owner}/{repo}: response wasn't valid JSON — stopping.")
                break
            if not prs:
                if page == 1:
                    print(f"[pr_descriptions] {owner}/{repo}: 0 PRs found (API call "
                          f"succeeded — this repo genuinely has none, e.g. a fresh "
                          f"fork with no PRs opened against it, as opposed to a "
                          f"failed fetch).")
                break

            for pr in prs:
                title = pr.get("title", "")
                body = pr.get("body") or ""
                content = f"PR #{pr.get('number')}: {title}\n{body}".strip()
                if content:
                    chunks.append({
                        "source_type": "pr_description",
                        "file_path": None,
                        "content": content,
                        "graph_node_id": None,
                        "metadata": {"pr_number": pr.get("number"), "state": pr.get("state")},
                    })
                fetched += 1
                if fetched >= MAX_PRS:
                    break

            page += 1

        return chunks

    @staticmethod
    def _parse_github_owner_repo(git_url: str) -> tuple[str, str] | None:
        """Extracts (owner, repo) from a github.com URL — HTTPS clone
        URLs with or without a trailing .git. Returns None for anything
        else (a local filesystem path, a different git host); the caller
        treats that as 'nothing to fetch', not an error."""
        match = re.match(
            r"https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$",
            git_url.strip(),
        )
        if match is None:
            return None
        return match.group(1), match.group(2)

    # --- file discovery ------------------------------------------------

    def _walk_files(self, local_path: str):
        try:
            result = subprocess.run(
                ["git", "-C", local_path, "ls-files"],
                check=True, capture_output=True, text=True,
            )
            for rel_path in result.stdout.splitlines():
                if rel_path:
                    yield rel_path
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        for dirpath, dirs, files in os.walk(local_path):
            dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
            for fname in files:
                yield os.path.relpath(os.path.join(dirpath, fname), local_path)

    # --- embedding + persistence ----------------------------------------

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i:i + EMBED_BATCH_SIZE]
            response = self._embed_batch_with_retry(batch)
            for item in response.embeddings:
                values = list(item.values)
                if len(values) != EMBEDDING_DIM:
                    raise ValueError(
                        f"Embedding dim mismatch: got {len(values)}, "
                        f"schema expects {EMBEDDING_DIM} (db/schema.sql "
                        f"code_chunks.embedding, config.EMBEDDING_DIM). "
                        f"output_dimensionality was requested as {EMBEDDING_DIM} — "
                        f"if this still mismatches, the model may not support "
                        f"that dimensionality; check current Gemini embedding docs."
                    )
                embeddings.append(values)
        return embeddings

    def _throttle_for_rate_limit(self) -> None:
        """Sliding-window self-throttle: before each request, drop
        timestamps older than the window, and if we're already at the
        per-minute cap, sleep just long enough for the oldest request in
        the window to age out. This is the primary defense against
        hitting the quota — proactive pacing rather than firing
        everything immediately and only reacting after a 429."""
        now = time.monotonic()
        self._request_timestamps = [
            t for t in self._request_timestamps if now - t < RATE_LIMIT_WINDOW_SECONDS
        ]
        if len(self._request_timestamps) >= EMBED_REQUESTS_PER_MINUTE:
            sleep_for = RATE_LIMIT_WINDOW_SECONDS - (now - self._request_timestamps[0]) + 0.5
            if sleep_for > 0:
                time.sleep(sleep_for)
            now = time.monotonic()
            self._request_timestamps = [
                t for t in self._request_timestamps if now - t < RATE_LIMIT_WINDOW_SECONDS
            ]
        self._request_timestamps.append(now)

    def _embed_batch_with_retry(self, batch: list[str]):
        """Throttles before every attempt, and retries specifically on a
        rate-limit error (detected by message content — "RESOURCE_EXHAUSTED"
        or "429" — rather than a specific exception class, since that's
        stable across google-genai SDK versions in a way an internal
        error-class layout might not be). Any OTHER error propagates
        immediately — retrying a non-rate-limit failure would just mask
        a real bug behind a delay."""
        last_error = None
        for attempt in range(MAX_RATE_LIMIT_RETRIES):
            self._throttle_for_rate_limit()
            try:
                return self._client.models.embed_content(
                    model=EMBEDDING_MODEL_NAME,
                    contents=batch,
                    config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM),
                )
            except Exception as e:
                message = str(e)
                if "RESOURCE_EXHAUSTED" not in message and "429" not in message:
                    raise
                last_error = e
                backoff = RATE_LIMIT_BASE_BACKOFF_SECONDS * (2 ** attempt)
                print(f"[rate limit] embed_content quota hit, retrying in "
                      f"{backoff:.0f}s (attempt {attempt + 1}/{MAX_RATE_LIMIT_RETRIES})...")
                time.sleep(backoff)
        raise last_error

    def _embed_and_persist(self, repo_id: str, chunks: list[dict]) -> None:
        if not chunks:
            return
        texts = [c["content"] for c in chunks]
        embeddings = self._embed_texts(texts)

        with connection() as conn:
            with conn.cursor() as cur:
                for chunk, embedding in zip(chunks, embeddings):
                    cur.execute(
                        """
                        INSERT INTO code_chunks
                            (id, repo_id, graph_node_id, source_type, file_path, content, embedding, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            str(uuid.uuid4()), repo_id, chunk.get("graph_node_id"),
                            chunk["source_type"], chunk["file_path"], chunk["content"],
                            embedding, json.dumps(chunk.get("metadata", {})),
                        ),
                    )

    def _clear_existing_chunks(self, repo_id: str) -> None:
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM code_chunks WHERE repo_id = %s", (repo_id,))