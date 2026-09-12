"""
Repo Registry — Phase 1.

Owns the lifecycle of registered repos: cloning, tracking in the `repos`
table, and (re)running RepoMapper to keep each repo's graph current.

Scope of this pass:
- sync() always does a full re-map, even if changed_files is passed in.
  True incremental sync (only re-parsing changed files) is deferred to
  Phase 7, where the Drift Detector's webhook trigger actually needs the
  speed — building it now would be premature before there's a caller that
  needs it.
- Webhook-triggered sync isn't wired up here either, for the same reason —
  this registry exposes sync() as a plain method; something calls it later.
- deregister() is a soft delete (status='inactive'), not a row delete, so
  the graph data for that repo (and any cross-repo edges pointing at it)
  stays intact — deliberately avoids dangling edges mid-trace, per the
  earlier design decision.
"""
import os
import stat
import shutil
import subprocess
import uuid
from dataclasses import dataclass

from config.settings import REPOS_CACHE_DIR
from db.connection import connection
from ingestion.repo_mapper import RepoMapper, persist_graph


def _rmtree_force(path: str) -> None:
    """shutil.rmtree that clears the read-only bit before removing —
    needed on Windows, where git marks files under .git/objects read-only
    and a plain rmtree fails with PermissionError (WinError 5) on them.
    No-op-safe on other platforms (chmod just succeeds trivially there)."""
    def _on_error(func, failed_path, _exc_info):
        os.chmod(failed_path, stat.S_IWRITE)
        func(failed_path)

    try:
        shutil.rmtree(path, onexc=lambda func, p, exc: _on_error(func, p, exc))
    except TypeError:
        # Python < 3.12 doesn't have onexc — fall back to the older onerror
        shutil.rmtree(path, onerror=_on_error)


@dataclass
class RepoRecord:
    id: str
    name: str
    git_url: str
    default_branch: str
    status: str
    local_path: str


class RepoRegistry:
    def _local_path(self, repo_id: str) -> str:
        return os.path.join(REPOS_CACHE_DIR, repo_id)

    @staticmethod
    def _run_git(args: list[str], error_context: str) -> None:
        """Wraps subprocess.run so a git failure surfaces its actual stderr
        instead of just an opaque exit-code CalledProcessError."""
        result = subprocess.run(args, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"{error_context} failed (exit {result.returncode}).\n"
                f"Command: {' '.join(args)}\n"
                f"stderr: {result.stderr.strip()}"
            )

    def register(self, name: str, git_url: str, default_branch: str = "main") -> str:
        repo_id = str(uuid.uuid4())
        local_path = self._local_path(repo_id)

        os.makedirs(REPOS_CACHE_DIR, exist_ok=True)
        if os.path.exists(local_path):
            # Leftover from a prior failed clone attempt — git clone refuses
            # to clone into a non-empty directory, which otherwise surfaces
            # as a confusing exit-128 with no clear cause.
            _rmtree_force(local_path)

        self._run_git(
            ["git", "clone", "--branch", default_branch, "--single-branch",
             git_url, local_path],
            error_context=f"git clone of {git_url}",
        )

        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO repos (id, name, git_url, default_branch, status)
                    VALUES (%s, %s, %s, %s, 'active')
                    """,
                    (repo_id, name, git_url, default_branch),
                )

        self._map_and_persist(repo_id, local_path)
        self._touch_last_synced(repo_id)
        return repo_id

    def deregister(self, repo_id: str) -> None:
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE repos SET status = 'inactive' WHERE id = %s",
                    (repo_id,),
                )

    def purge(self, repo_id: str) -> None:
        """Hard delete — for dev/testing iteration only, NOT the
        production path (use deregister() there). Removes the repos row
        (graph_nodes/graph_edges cascade-delete via FK) and the local
        clone directory, freeing the `name` unique constraint so the same
        repo can be re-registered from scratch."""
        record = self.get_repo(repo_id)
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM repos WHERE id = %s", (repo_id,))
        if record is not None and os.path.isdir(record.local_path):
            _rmtree_force(record.local_path)

    def get_repo_by_name(self, name: str) -> RepoRecord | None:
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, name, git_url, default_branch, status
                    FROM repos WHERE name = %s
                    """,
                    (name,),
                )
                row = cur.fetchone()
        if row is None:
            return None
        return RepoRecord(
            id=str(row[0]), name=row[1], git_url=row[2],
            default_branch=row[3], status=row[4],
            local_path=self._local_path(str(row[0])),
        )

    def sync(self, repo_id: str, changed_files: list[str] | None = None) -> None:
        """Full re-map for now — see module docstring. `changed_files` is
        accepted so callers (e.g. a future webhook handler) have a stable
        signature to code against, but it's currently unused."""
        record = self.get_repo(repo_id)
        if record is None:
            raise ValueError(f"No repo registered with id={repo_id}")
        if record.status != "active":
            raise ValueError(f"Repo {repo_id} is not active (status={record.status})")

        self._run_git(
            ["git", "-C", record.local_path, "pull", "origin", record.default_branch],
            error_context=f"git pull for repo {repo_id}",
        )

        self._clear_existing_graph(repo_id)
        self._map_and_persist(repo_id, record.local_path)
        self._touch_last_synced(repo_id)

    def get_repo(self, repo_id: str) -> RepoRecord | None:
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, name, git_url, default_branch, status
                    FROM repos WHERE id = %s
                    """,
                    (repo_id,),
                )
                row = cur.fetchone()
        if row is None:
            return None
        return RepoRecord(
            id=str(row[0]), name=row[1], git_url=row[2],
            default_branch=row[3], status=row[4],
            local_path=self._local_path(str(row[0])),
        )

    def list_active_repos(self) -> list[RepoRecord]:
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, name, git_url, default_branch, status FROM repos WHERE status = 'active'"
                )
                rows = cur.fetchall()
        return [
            RepoRecord(
                id=str(r[0]), name=r[1], git_url=r[2], default_branch=r[3],
                status=r[4], local_path=self._local_path(str(r[0])),
            )
            for r in rows
        ]

    # --- internals ---------------------------------------------------------

    def _map_and_persist(self, repo_id: str, local_path: str) -> None:
        mapper = RepoMapper(repo_id=repo_id, local_path=local_path)
        graph = mapper.build_graph()
        persist_graph(graph)

    def _clear_existing_graph(self, repo_id: str) -> None:
        """Deletes this repo's graph_nodes (graph_edges cascade via FK) so
        sync() doesn't duplicate nodes on re-run."""
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM graph_nodes WHERE repo_id = %s", (repo_id,))

    def _touch_last_synced(self, repo_id: str) -> None:
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE repos SET last_synced_at = now() WHERE id = %s",
                    (repo_id,),
                )