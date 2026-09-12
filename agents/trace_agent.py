"""
Trace Agent — Phase 4.

Iterative search -> read -> decide-to-go-deeper investigation across the
graph_edges Phase 1 built (CALLS, IMPORTS, DEFINES, and cross-repo
http_call/manual edges from CrossRepoLinker). This is the "follow the
reference to find the real answer" capability RepoMapper deliberately
punted on for cross-file constant resolution — see repo_mapper.py's
module docstring for that earlier decision. This agent is where that
kind of reasoning actually belongs: with retrieval available to weigh
options, not baked into static single-file analysis.

At each hop: fetch the current node's content (preferring an already-
embedded code_chunks row — same content the rest of the system already
uses — falling back to reading the exact lines from the repo's local
checkout if no chunk exists for that node), fetch its outgoing edges, and
ask the model whether the current context already answers the question or
which single edge is worth following next. Stops on: sufficient context
found, a dead end (no outgoing edges), a cycle, max_hops reached, or an
unparseable/invalid model decision (fails safely rather than guessing a
hop).

Real-testing finding worth documenting (not hypothetical): a
FUNCTION/CLASS node very often has ZERO direct outgoing edges even when
its behavior clearly depends on another file — because RepoMapper's
CALLS edges only capture same-file bare-identifier calls (not attribute/
method calls like `some_object.method()`, and not cross-file calls to an
imported name, by design), while IMPORTS edges live on the FILE node, not
the function node within it. A function-level starting point structurally
can't see its own file's imports. _fetch_outgoing_edges works around this
by folding in the enclosing file's own edges as additional, clearly
file-level-labeled candidate hops whenever the node's own direct edges
don't cover something — so a function isn't a dead end just because the
relevant reference happens to be an attribute call or a cross-file import
rather than a same-file bare call.
"""
import os
import json

from config.settings import AgentRole
from models.llm_clients import get_client_for_role
from db.connection import connection
from ingestion.repo_registry import RepoRegistry


class TraceAgent:
    role = AgentRole.TRACE_AGENT

    def __init__(self):
        self._registry = RepoRegistry()

    def trace(self, starting_node_id: str, question: str, max_hops: int = 5) -> dict:
        """Returns {"trace": [{"hop", "node_id", "qualified_name",
        "file_path", "repo_id", "content", "metadata"}, ...],
        "hops_taken": int, "stopped_reason": str}."""
        visited: set[str] = set()
        current_id = starting_node_id
        trace_steps: list[dict] = []
        stopped_reason = "max_hops_reached"

        for hop in range(max_hops):
            if current_id in visited:
                stopped_reason = "cycle_detected"
                break
            visited.add(current_id)

            content, node_info = self._fetch_node_content(current_id)
            if node_info is None:
                stopped_reason = "node_not_found"
                break

            edges = self._fetch_outgoing_edges(current_id)
            trace_steps.append({
                "hop": hop,
                "node_id": current_id,
                "qualified_name": node_info["qualified_name"],
                "file_path": node_info["file_path"],
                "repo_id": node_info["repo_id"],
                "content": content,
                "metadata": node_info["metadata"],
            })

            if not edges:
                stopped_reason = "dead_end_no_further_references"
                break

            decision = self._decide_next_hop(question, node_info, content, edges)
            if decision is None:
                stopped_reason = "decision_unparseable_stopping_safely"
                break
            if decision.get("sufficient"):
                stopped_reason = "sufficient_context_found"
                break

            next_id = decision.get("follow_target_id")
            valid_targets = {e["target_node_id"] for e in edges}
            if next_id not in valid_targets:
                # Model named a hop that doesn't exist among the options it
                # was given — stop rather than guess which one it meant.
                stopped_reason = "invalid_hop_choice_stopping_safely"
                break
            current_id = next_id

        return {
            "trace": trace_steps,
            "hops_taken": len(trace_steps),
            "stopped_reason": stopped_reason,
        }

    def trace_to_chunks(self, trace_result: dict) -> list[dict]:
        """Converts trace steps into the same shape AnswerSynthesizer
        expects from retrieved_chunks, so a traced investigation can be
        synthesized into an answer via the exact same AnswerSynthesizer
        used for single-hop retrieval — no separate synthesis path
        needed just because the context came from tracing instead of
        search."""
        return [
            {
                "source_type": "code",
                "file_path": step["file_path"],
                "content": step["content"],
                "metadata": step["metadata"],
                "similarity": None,
            }
            for step in trace_result["trace"]
        ]

    # --- node/edge lookups ------------------------------------------

    def _fetch_node_content(self, node_id: str) -> tuple[str | None, dict | None]:
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT qualified_name, file_path, repo_id, node_type, metadata "
                    "FROM graph_nodes WHERE id = %s",
                    (node_id,),
                )
                row = cur.fetchone()
        if row is None:
            return None, None
        qualified_name, file_path, repo_id, node_type, metadata = row
        node_info = {
            "qualified_name": qualified_name,
            "file_path": file_path,
            "repo_id": str(repo_id),
            "node_type": node_type,
            "metadata": metadata or {},
        }

        # Prefer an already-embedded chunk for this exact node — same
        # content the rest of the system already uses, no extra file I/O.
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT content FROM code_chunks WHERE graph_node_id = %s LIMIT 1",
                    (node_id,),
                )
                chunk_row = cur.fetchone()
        if chunk_row is not None:
            return chunk_row[0], node_info

        # Fallback: read the exact lines directly from the repo's local
        # checkout, if we have a line range and the clone still exists
        # (e.g. this node's repo hasn't been through Semantic Indexing
        # yet, or its chunk was windowed rather than precise).
        start_line = node_info["metadata"].get("start_line")
        end_line = node_info["metadata"].get("end_line")
        if start_line and end_line:
            record = self._registry.get_repo(node_info["repo_id"])
            if record is not None:
                abs_path = os.path.join(record.local_path, file_path)
                try:
                    with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                        lines = f.readlines()
                    content = "".join(lines[start_line - 1:end_line])
                    return content, node_info
                except OSError:
                    pass

        return f"(no content available for {qualified_name})", node_info

    def _fetch_outgoing_edges(self, node_id: str) -> list[dict]:
        """Direct edges for this exact node, plus (if this is a
        function/class) the enclosing FILE node's own outgoing edges
        folded in as additional candidates — labeled via_file_level_import
        so the model can weigh them as coarser/lower-precision than a
        direct reference. See module docstring for why this matters: a
        function's own direct edges very often don't cover the thing
        that actually answers the question."""
        direct_edges = self._fetch_direct_outgoing_edges(node_id)
        file_edges = self._fetch_enclosing_file_edges(node_id)
        print(f"[trace debug] node_id={node_id}: {len(direct_edges)} direct edges, "
              f"{len(file_edges)} enclosing-file edges found")

        seen_targets = {e["target_node_id"] for e in direct_edges}
        for e in file_edges:
            if e["target_node_id"] not in seen_targets:
                direct_edges.append(e)
                seen_targets.add(e["target_node_id"])
        print(f"[trace debug] {len(direct_edges)} total edges after merge")
        return direct_edges

    def _fetch_direct_outgoing_edges(self, node_id: str) -> list[dict]:
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT ge.target_node_id, ge.edge_type, ge.is_cross_repo, ge.confidence,
                           gn.qualified_name, gn.file_path, r.name
                    FROM graph_edges ge
                    JOIN graph_nodes gn ON ge.target_node_id = gn.id
                    JOIN repos r ON gn.repo_id = r.id
                    WHERE ge.source_node_id = %s
                    """,
                    (node_id,),
                )
                rows = cur.fetchall()
        return [
            {
                "target_node_id": str(r[0]),
                "edge_type": r[1],
                "is_cross_repo": r[2],
                "confidence": r[3],
                "target_qualified_name": r[4],
                "target_file_path": r[5],
                "target_repo_name": r[6],
                "via_file_level_import": False,
            }
            for r in rows
        ]

    def _fetch_enclosing_file_edges(self, node_id: str) -> list[dict]:
        """For a function/class node, finds its enclosing FILE node (same
        repo_id + file_path) and returns THAT node's outgoing edges,
        marked via_file_level_import=True. Returns [] for a node that's
        already a file (or a module-level marker node), since there's
        nothing coarser to fold in."""
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT repo_id, file_path, node_type FROM graph_nodes WHERE id = %s",
                    (node_id,),
                )
                row = cur.fetchone()
        if row is None:
            print(f"[trace debug] _fetch_enclosing_file_edges: node {node_id} not found at all")
            return []
        repo_id, file_path, node_type = row
        if node_type in ("file", "module"):
            print(f"[trace debug] _fetch_enclosing_file_edges: node {node_id} is already a "
                  f"'{node_type}' node, nothing coarser to fold in")
            return []

        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id FROM graph_nodes WHERE repo_id = %s AND file_path = %s AND node_type = 'file'",
                    (repo_id, file_path),
                )
                file_row = cur.fetchone()
        if file_row is None:
            print(f"[trace debug] _fetch_enclosing_file_edges: no FILE node found for "
                  f"repo_id={repo_id} file_path={file_path!r}")
            return []
        file_node_id = str(file_row[0])
        if file_node_id == node_id:
            return []

        edges = self._fetch_direct_outgoing_edges(file_node_id)
        print(f"[trace debug] _fetch_enclosing_file_edges: enclosing file node "
              f"{file_node_id} ({file_path!r}) has {len(edges)} direct edges")
        for e in edges:
            e["via_file_level_import"] = True
        return edges

    # --- hop decision ---------------------------------------------------

    def _decide_next_hop(self, question: str, node_info: dict, content: str, edges: list[dict]) -> dict | None:
        hop_options = "\n".join(
            f"- id={e['target_node_id']}: {e['edge_type']}"
            f"{' [cross-repo]' if e['is_cross_repo'] else ''}"
            f"{' (inferred, lower confidence)' if e['confidence'] == 'inferred' else ''}"
            f"{' (via this file imports — not a direct reference from the specific code above)' if e.get('via_file_level_import') else ''}"
            f" -> {e['target_qualified_name']} in {e['target_repo_name']} ({e['target_file_path']})"
            for e in edges
        )
        system_prompt = (
            "You are investigating a codebase to answer a question by "
            "following code references (imports, function calls, "
            "cross-repo HTTP calls) one hop at a time. You'll see the "
            "current code and a list of possible next hops. Decide whether "
            "the current information is SUFFICIENT to answer the question, "
            "or which SINGLE hop to follow next for more context. Prefer a "
            "direct, confirmed hop over an inferred or file-level-import "
            "one unless the current code clearly references something that "
            "only a file-level import would explain.\n"
            "Respond ONLY with JSON, no other text:\n"
            '{"sufficient": true|false, "follow_target_id": "<id>"|null, "reasoning": "brief"}'
        )
        user_prompt = (
            f"Question: {question}\n\n"
            f"Current location: {node_info['qualified_name']} ({node_info['file_path']})\n\n"
            f"Code:\n{content}\n\n"
            f"Possible next hops:\n{hop_options}"
        )
        client = get_client_for_role(self.role)
        raw = client.complete(system_prompt, user_prompt)
        return self._safe_parse_json(raw)

    @staticmethod
    def _safe_parse_json(raw: str) -> dict | None:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
        try:
            return json.loads(cleaned.strip())
        except (json.JSONDecodeError, TypeError):
            return None