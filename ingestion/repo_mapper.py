"""
Repo Mapper — Phase 1.

Walks a single repo, parses Python and TypeScript/JavaScript files with
tree-sitter, and produces a RepoGraph: file/function/class nodes, plus
IMPORTS and CALLS edges.

Scope of this pass (deliberately limited, see TODOs):
- Import resolution handles relative imports (`from .foo import bar`),
  absolute package-style imports (`from agents.tools import x`), and bare
  same-directory imports (`from resilience import x`) — see
  _guess_local_import_candidates. External package imports are recorded
  in a file node's metadata but do not get their own graph_edges row.
- Call resolution only resolves calls to functions/classes defined in the
  SAME file. Cross-file call resolution is left to the Trace Agent (Phase 4),
  which has retrieval available to disambiguate — doing it here with pure
  static analysis would be unreliable for dynamic languages like Python/JS.
- Cross-repo edges are explicitly out of scope for this file — see
  ingestion/cross_repo_linker.py. However, this module DOES capture
  HTTP-call-site string literals (file_node.metadata["call_site_literals"])
  during the same AST pass used for everything else, so the linker doesn't
  need a second raw-text scan of every file. Detection is by ARGUMENT
  SHAPE, not by callee name — see _extract_http_call_literal's docstring
  for why (short version: hardcoding known HTTP client library names
  doesn't scale across languages/libraries). The first positional argument
  is resolved through string literals, f-strings/template literals
  (interpolations become '{}' wildcards, resolving known constants and
  env-var defaults where possible), string concatenation, and simple
  identifiers (branch-aware, see _resolve_identifier_literal) — then kept
  only if the resolved value actually looks like a URL or absolute path.
- File discovery prefers `git ls-files` (fast, respects .gitignore) and
  falls back to a manual os.walk if the repo isn't a git checkout.
- DELIBERATE BOUNDARY (decided after real-world testing, not an oversight):
  literal/identifier resolution — everything above — is scoped to a
  SINGLE FILE's AST. A constant imported from another file in the same
  repo (`from data.urls import BASE_URL`, then `BASE_URL + url` used in
  an HTTP call) will NOT resolve, even though the value technically
  exists in the repo. Extending this module to resolve cross-file
  constants was considered and explicitly rejected: it would mean
  resolution correctness depends on another file's content (new failure
  modes — what if that file fails to parse? what if there's a genuine
  cross-file cycle?), for a problem the Trace Agent (Phase 4) is already
  designed to solve properly, with retrieval, rather than a narrower
  static reimplementation of the same idea done early. A call site whose
  literal can't be resolved for this reason simply produces no
  call_site_literals entry — same graceful degradation as any other
  unresolvable case in this module, not a bug to chase.
"""
import sys
import os
# Dynamically add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv

# Load variables from the .env file
load_dotenv()

import uuid
import json
import subprocess
from dataclasses import dataclass, field
from enum import Enum

from tree_sitter import Language, Parser
import tree_sitter_python as tspython
import tree_sitter_typescript as tstypescript

from db.connection import connection


class NodeType(str, Enum):
    MODULE = "module"
    FILE = "file"
    FUNCTION = "function"
    CLASS = "class"
    ENDPOINT = "endpoint"  # populated later when OpenAPI/proto parsing lands


class EdgeType(str, Enum):
    IMPORTS = "imports"
    CALLS = "calls"
    DEFINES = "defines"  # file -> function/class it contains


@dataclass
class RepoNode:
    """Single-repo node — matches the `graph_nodes` table shape."""
    id: str
    node_type: NodeType
    qualified_name: str
    file_path: str
    metadata: dict = field(default_factory=dict)


@dataclass
class RepoEdge:
    """Single-repo edge — cross-repo edges are a separate concern, added by
    ingestion/cross_repo_linker.py, not by this module."""
    id: str
    source_node_id: str
    target_node_id: str
    edge_type: EdgeType


@dataclass
class RepoGraph:
    """Output format for a single repo's structural map."""
    repo_id: str
    nodes: list[RepoNode] = field(default_factory=list)
    edges: list[RepoEdge] = field(default_factory=list)


SUPPORTED_EXTENSIONS = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "javascript",
}

IGNORED_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", ".mypy_cache", ".pytest_cache",
}


class RepoMapper:
    def __init__(self, repo_id: str, local_path: str):
        self.repo_id = repo_id
        self.local_path = local_path
        self._parsers = self._build_parsers()
        # Cycle guard for literal resolution — see _extract_literal_any_shape.
        # Tracks (start_byte, end_byte) rather than Python object id(), since
        # tree-sitter bindings aren't guaranteed to hand back the same
        # wrapper object for the "same" tree position on repeated access —
        # byte offsets are a stable, position-based identity that doesn't
        # depend on that.
        self._resolving_stack: set[tuple[int, int]] = set()

    def _build_parsers(self) -> dict[str, Parser]:
        py_lang = Language(tspython.language())
        ts_lang = Language(tstypescript.language_typescript())
        tsx_lang = Language(tstypescript.language_tsx())

        parsers = {}
        for lang_key, lang in [
            ("python", py_lang),
            ("typescript", ts_lang),
            ("tsx", tsx_lang),
            ("javascript", ts_lang),  # JS parses fine with the TS grammar
        ]:
            p = Parser(lang)
            parsers[lang_key] = p
        return parsers

    @staticmethod
    def _line_range_metadata(node) -> dict:
        """Captures a function/class node's exact 1-indexed line span
        (tree-sitter's start_point/end_point are 0-indexed (row, column)
        tuples). Stored in RepoNode.metadata rather than as new dataclass
        fields or DB columns — metadata is already a JSONB column that
        exists for exactly this kind of thing, so no schema change is
        needed. This is what lets the Semantic Indexer (Phase 2) chunk
        by exact function/class boundaries instead of guessing with
        fixed-size line windows."""
        return {
            "start_line": node.start_point[0] + 1,
            "end_line": node.end_point[0] + 1,
        }

    # --- literal extraction helpers ----------------------------------

    @staticmethod
    def _strip_string_literal(raw: str) -> str:
        """Strips quotes and common string prefixes (b/f/r/u) from a
        tree-sitter 'string' node's raw text."""
        stripped = raw.lstrip("bBfFrRuU")
        if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in ("'", '"'):
            return stripped[1:-1]
        return stripped

    def _extract_string_with_wildcards(self, node, text_fn, root, is_python: bool) -> str | None:
        """Handles a Python 'string' node, including f-strings. Plain
        strings are returned as-is (quotes stripped). For each interpolated
        {expr} in an f-string, the inner expression is resolved via the
        same general dispatcher (_extract_literal_any_shape) used
        everywhere else — not just for bare identifiers, but anything it
        knows how to resolve (a constant, a concatenation, an env-var
        default). This matters: wildcarding a resolvable host constant
        would turn "http://svc:8000/orders/{id}" into "{}/orders/{}",
        destroying the path structure a spec-match needs. Whatever can't
        be resolved becomes a literal '{}' placeholder, which the
        Cross-Repo Linker's path normalizer already treats as a wildcard
        segment (same rule it uses for {id}-style path params).

        Routing through the shared dispatcher (rather than calling
        identifier-resolution directly) is also what makes this safe from
        cycles — see _extract_literal_any_shape's cycle guard, which only
        works if every resolution path funnels through that one entry
        point rather than some paths bypassing it."""
        if node.type != "string":
            return None
        if not any(c.type == "interpolation" for c in node.children):
            return self._strip_string_literal(text_fn(node))
        parts = []
        for child in node.children:
            if child.type == "interpolation":
                expr_node = child.child_by_field_name("expression")
                if expr_node is None and child.named_children:
                    expr_node = child.named_children[0]
                resolved = None
                if expr_node is not None:
                    resolved = self._extract_literal_any_shape(expr_node, text_fn, root, is_python)
                parts.append(resolved if resolved is not None else "{}")
            elif child.type in ("string_start", "string_end"):
                continue
            else:
                parts.append(text_fn(child))
        return "".join(parts)

    def _extract_template_with_wildcards(self, node, text_fn, root, is_python: bool) -> str | None:
        """TS/JS equivalent of the above for template literals:
        `http://${HOST}/webhook/${path}` -> resolves HOST if it's a known
        constant, wildcards ${path} if it isn't. Same dispatcher-routing
        reasoning as _extract_string_with_wildcards above."""
        if node.type != "template_string":
            return None
        parts = []
        for child in node.children:
            if child.type == "template_substitution":
                expr_node = None
                if child.named_children:
                    expr_node = child.named_children[0]
                resolved = None
                if expr_node is not None:
                    resolved = self._extract_literal_any_shape(expr_node, text_fn, root, is_python)
                parts.append(resolved if resolved is not None else "{}")
            elif child.type == "`":
                continue
            else:
                parts.append(text_fn(child))
        return "".join(parts)

    def _extract_literal_any_shape(self, node, text_fn, root, is_python: bool) -> str | None:
        """Single dispatch point for 'resolve this node to a literal, if
        possible' — used everywhere a literal might be needed (call
        arguments, assignment right-hand sides, concatenation operands).
        Centralizing this is what let string-concatenation support
        slot in as one new case instead of being re-implemented at every
        call site — and it's also the single place a general cycle guard
        can live, rather than needing bespoke handling for every specific
        AST shape that could cause a resolution cycle (direct
        self-reference, mutual two-variable reference, longer chains —
        the exact shape doesn't matter to this guard, only whether we're
        being asked to resolve something already in progress higher up
        the call stack). That's a deliberate strategy shift from patching
        individual patterns as they're discovered (unsustainable — there's
        always another one) to making the whole resolution system safe
        against ANY cycle by construction."""
        node_key = (node.start_byte, node.end_byte)
        if node_key in self._resolving_stack:
            return None  # cycle — this exact position is already being resolved higher up
        self._resolving_stack.add(node_key)
        try:
            return self._dispatch_literal_shape(node, text_fn, root, is_python)
        finally:
            self._resolving_stack.discard(node_key)

    def _dispatch_literal_shape(self, node, text_fn, root, is_python: bool) -> str | None:
        """The actual shape dispatch, separated from the cycle-guard
        wrapper in _extract_literal_any_shape for clarity."""
        if node.type == "string":
            if is_python:
                return self._extract_string_with_wildcards(node, text_fn, root, is_python)
            return self._strip_string_literal(text_fn(node))
        if node.type == "template_string":
            return self._extract_template_with_wildcards(node, text_fn, root, is_python)
        if node.type == "identifier":
            return self._resolve_identifier_literal(text_fn(node), node, root, text_fn, is_python)
        if is_python and node.type == "binary_operator":
            op_field = node.child_by_field_name("operator")
            if op_field is not None and text_fn(op_field) == "+":
                return self._extract_concat_with_wildcards(node, text_fn, root, is_python)
            return None
        if not is_python and node.type == "binary_expression":
            op_field = node.child_by_field_name("operator")
            if op_field is None:
                return None
            op_text = text_fn(op_field)
            if op_text == "+":
                return self._extract_concat_with_wildcards(node, text_fn, root, is_python)
            if op_text in ("||", "??"):
                return self._extract_env_default_literal(node, text_fn, root, is_python)
            return None
        if is_python and node.type == "call":
            return self._extract_env_default_literal(node, text_fn, root, is_python)
        return None

    def _extract_concat_with_wildcards(self, node, text_fn, root, is_python: bool) -> str | None:
        """Handles string concatenation building a URL/path piecewise —
        e.g. `BASE_URL + "/pet/" + str(pet_id)` (Python) or
        `BASE_URL + '/pet/' + id` (JS). This is a different AST shape
        entirely from f-strings/template literals (a binary_operator /
        binary_expression chain, not a single string node with
        interpolations), and is at least as common in real code — string
        concatenation predates f-strings and never went away.

        Each operand is resolved recursively via
        `_extract_literal_any_shape` (so a 3+ term chain like
        `A + B + C` — which parses as nested binary nodes — unwinds
        naturally through recursion on the left operand). An operand that
        can't be resolved (a function call like `str(x)`, a number, an
        f-string with dynamic content) becomes a `{}` wildcard placeholder
        rather than aborting the whole expression — same philosophy as
        unresolvable f-string interpolations elsewhere in this module.
        Returns None only if NEITHER side contributed anything real,
        since an all-wildcard result has nothing for the Cross-Repo
        Linker to match against."""
        left = node.child_by_field_name("left")
        right = node.child_by_field_name("right")
        if left is None or right is None:
            return None

        left_part = self._extract_literal_any_shape(left, text_fn, root, is_python)
        if left_part is None:
            left_part = "{}"
        right_part = self._extract_literal_any_shape(right, text_fn, root, is_python)
        if right_part is None:
            right_part = "{}"

        if left_part == "{}" and right_part == "{}":
            return None
        return left_part + right_part

    def _extract_http_call_literal(self, call_node, text_fn, root, is_python: bool) -> str | None:
        """Detects HTTP call sites by ARGUMENT SHAPE, not by callee name.

        Earlier versions of this gated on a hardcoded allowlist of known
        HTTP client object/method names (requests.get, axios.post,
        fetch(...), etc.) — which meant every unlisted library (superagent,
        got, ky, a custom internal wrapper like call_n8n_webhook, anything
        in a language we don't special-case) was silently invisible. That
        doesn't scale: there's no way to enumerate every HTTP client in
        every language ahead of time, and a product that needs its source
        edited every time it meets an unfamiliar codebase isn't durable.

        Instead: resolve the first positional argument through whatever
        shape it happens to be (string literal, f-string/template literal,
        identifier, or string concatenation — see
        `_extract_literal_any_shape`) regardless of what the call itself
        looks like, then check whether the RESOLVED VALUE looks like a URL
        or an absolute path (`_looks_like_url_or_path`). This is library-
        and even language-agnostic — it works identically for
        `requests.get`, `superagent.get`, `fetch`, or a hand-rolled
        `call_n8n_webhook`, because none of them matter; only the shape of
        the data being passed does.

        Trade-off, stated plainly: this is a broader net, so an unrelated
        call that happens to pass a URL-shaped string (e.g. logging a path,
        constructing a File object) will also get captured as a candidate
        literal. That's an acceptable cost, not a correctness problem —
        Cross-Repo Linker's confirmed/inferred matching still requires that
        literal to actually match a real spec endpoint or a real repo name
        before it becomes an edge, so a false-positive-shaped literal from
        an irrelevant call just sits unused rather than producing a wrong
        link."""
        args_field = call_node.child_by_field_name("arguments")
        if args_field is None:
            return None
        positional_args = [c for c in args_field.named_children if c.type not in ("keyword_argument",)]
        if not positional_args:
            return None
        first_arg = positional_args[0]

        literal = self._extract_literal_any_shape(first_arg, text_fn, root, is_python)

        if literal is None or not self._looks_like_url_or_path(literal):
            return None
        return literal

    @staticmethod
    def _looks_like_url_or_path(literal: str) -> bool:
        """The one remaining filter: does the resolved value actually look
        like something you'd make a network call to? Deliberately simple
        and shape-based (scheme prefix, or a leading '/' followed by real
        content) rather than another name-based allowlist."""
        if literal.startswith("http://") or literal.startswith("https://"):
            return True
        if literal.startswith("/") and len(literal) > 1:
            return True
        return False

    # --- branch-aware identifier resolution ---------------------------

    @staticmethod
    def _enclosing_function(node, root, is_python: bool):
        """Walks up from `node` to find its nearest enclosing function —
        the scope boundary for variable resolution. Returns None if the
        call is at module/top level (caller then searches the whole file).

        Compares nodes via `.id` (tree-sitter's own stable per-position
        identifier), never Python `is`/`id()` — tree-sitter's bindings do
        not guarantee that traversing to the same tree position twice
        returns the same Python wrapper object, so identity comparison is
        unreliable and was the actual root cause of a recursion bug found
        via real-world testing (see _is_ancestor)."""
        function_types = ("function_definition",) if is_python else (
            "function_declaration", "arrow_function", "function_expression", "method_definition",
        )
        cur = node.parent
        while cur is not None and cur.id != root.id:
            if cur.type in function_types:
                return cur
            cur = cur.parent
        return None

    @staticmethod
    def _same_node(a, b) -> bool:
        """Stable node-identity comparison via tree-sitter's own `.id`
        (see _enclosing_function's docstring for why this exists instead
        of `is`/`==`/`id()`). Treats two Nones as equal, and a None vs a
        real node as unequal."""
        if a is None or b is None:
            return a is b
        return a.id == b.id

    @staticmethod
    def _branch_label(parent, child) -> str | None:
        """For a direct parent/child pair, returns which mutually-exclusive
        branch `child` represents under `parent` (e.g. 'consequence' vs
        'alternative' for an if/else), or None if `parent` isn't a
        branching construct / `child` isn't one of its exclusive arms
        (e.g. a 'finally' block, which always runs and so isn't exclusive
        with anything)."""
        if parent.type in ("if_statement", "elif_clause"):
            if RepoMapper._same_node(child, parent.child_by_field_name("consequence")):
                return "consequence"
            if RepoMapper._same_node(child, parent.child_by_field_name("alternative")):
                return "alternative"
            if child.type in ("elif_clause", "else_clause"):
                return "alternative"
            return None
        if parent.type == "try_statement":
            if child.type == "except_clause":
                return "except"
            if child.type == "finally_clause":
                return None  # runs regardless — not exclusive with anything
            return "try"
        return None

    @staticmethod
    def _mutually_exclusive(node_a, node_b, stop_at) -> bool:
        """Returns True if node_a and node_b sit in different, mutually
        exclusive branches of a shared if/elif/else or try/except ancestor
        found between them and `stop_at` (their common enclosing function,
        exclusive). This is the core of the branch-awareness: two
        assignments (or an assignment and a call) in sibling if/else
        branches genuinely cannot both apply to the same execution, so
        resolution treats them as incompatible rather than guessing.

        All node comparisons use `.id`, not `is`/`id()` — see
        _enclosing_function's docstring."""
        def ancestor_chain(n):
            chain = []
            cur = n
            while cur is not None and cur.id != stop_at.id:
                parent = cur.parent
                if parent is None:
                    break
                chain.append((parent, cur))
                if parent.id == stop_at.id:
                    break
                cur = parent
            return chain

        chain_a = ancestor_chain(node_a)
        chain_b = ancestor_chain(node_b)
        by_parent_a: dict[int, list] = {}
        for parent, child in chain_a:
            by_parent_a.setdefault(parent.id, []).append(child)

        for parent, child_b in chain_b:
            for child_a in by_parent_a.get(parent.id, []):
                if child_a.id == child_b.id:
                    continue
                branch_a = RepoMapper._branch_label(parent, child_a)
                branch_b = RepoMapper._branch_label(parent, child_b)
                if branch_a is not None and branch_b is not None and branch_a != branch_b:
                    return True
        return False

    def _resolve_identifier_literal(
        self, name: str, reference_node, root, text_fn, is_python: bool
    ) -> str | None:
        """Resolves an identifier reference (e.g. `url` in
        `requests.post(url, ...)`, or `ORDERS_BASE_URL` inside an
        f-string interpolation) back to a string/f-string literal assigned
        to it earlier.

        Two-tier, mirroring real lexical scoping instead of just searching
        the whole file flatly:
        1. Search the enclosing function first. If the name is assigned
           ANYWHERE in that function (even ambiguously), that's the
           answer — Python treats a name assigned anywhere in a function
           as local to it, so it never falls through to a module-level
           global of the same name, even if the local assignment turns
           out to be unresolvable here.
        2. Only if the name is never locally assigned does this fall back
           to module-level scope — and that search deliberately does NOT
           descend into other functions' bodies, so it only ever matches
           genuine module-level constants (like `ORDERS_BASE_URL = "..."`
           sitting at the top of a file), never an unrelated local
           variable of the same name in some other function.

        Within whichever scope applies (Option B, refuse rather than
        guess):
        - Only assignments textually before `reference_node` count.
        - An assignment in a branch mutually exclusive with the reference
          (e.g. assigned inside an `if` the reference isn't part of) is
          dropped — it could never actually be the value in effect there.
        - If what's left all agrees on one literal, that's unambiguous.
          If they disagree and are themselves mutually exclusive with each
          other (genuinely "depends which branch ran"), returns None. If
          they disagree but aren't mutually exclusive (straight-line
          reassignment), the nearest preceding one wins.

        Does not model loops (a variable set differently across loop
        iterations) — out of scope for this pass, same as the rest of this
        module's static analysis.
        """
        enclosing = self._enclosing_function(reference_node, root, is_python)
        if enclosing is not None:
            local_result, had_local_candidates = self._resolve_in_scope(
                name, reference_node, enclosing, text_fn, is_python,
                skip_nested_functions=False, true_root=root,
            )
            if had_local_candidates:
                return local_result  # local shadows global even if ambiguous — no fallback

        global_result, _ = self._resolve_in_scope(
            name, reference_node, root, text_fn, is_python,
            skip_nested_functions=True, true_root=root,
        )
        return global_result

    @staticmethod
    def _is_env_getenv_call(fn_field, text_fn) -> bool:
        """Recognizes os.getenv(...) and os.environ.get(...) call shapes."""
        if fn_field is None or fn_field.type != "attribute":
            return False
        obj = fn_field.child_by_field_name("object")
        attr = fn_field.child_by_field_name("attribute")
        if obj is None or attr is None:
            return False
        attr_name = text_fn(attr)
        if obj.type == "identifier" and text_fn(obj) == "os" and attr_name == "getenv":
            return True
        if obj.type == "attribute" and attr_name == "get":
            inner_obj = obj.child_by_field_name("object")
            inner_attr = obj.child_by_field_name("attribute")
            if (
                inner_obj is not None and inner_attr is not None
                and inner_obj.type == "identifier" and text_fn(inner_obj) == "os"
                and text_fn(inner_attr) == "environ"
            ):
                return True
        return False

    def _extract_env_default_literal(self, node, text_fn, root, is_python: bool) -> str | None:
        """Extracts the literal DEFAULT value from an environment-variable
        read, when one is present — the actual runtime env var value is of
        course unknowable statically, but its documented/fallback default
        is a genuinely useful signal and an extremely common pattern for
        backend base URLs specifically:
        - Python: os.getenv("NAME", "default") / os.environ.get("NAME", "default")
        - TS/JS:  process.env.NAME || "default"  /  process.env.NAME ?? "default"
        Returns None if there's no literal default (e.g. a bare
        `os.getenv("NAME")` or `process.env.NAME` with nothing to fall
        back on) — that case correctly stays unresolved rather than
        guessing, same as everywhere else in this module."""
        if is_python:
            if node.type != "call":
                return None
            fn_field = node.child_by_field_name("function")
            if not self._is_env_getenv_call(fn_field, text_fn):
                return None
            args_field = node.child_by_field_name("arguments")
            if args_field is None:
                return None
            positional = [c for c in args_field.named_children if c.type != "keyword_argument"]
            if len(positional) < 2:
                return None  # no default supplied
            return self._extract_literal_any_shape(positional[1], text_fn, root, is_python)
        else:
            if node.type != "binary_expression":
                return None
            op_field = node.child_by_field_name("operator")
            left = node.child_by_field_name("left")
            right = node.child_by_field_name("right")
            if op_field is None or left is None or right is None:
                return None
            if text_fn(op_field) not in ("||", "??"):
                return None
            if not text_fn(left).replace(" ", "").startswith("process.env."):
                return None
            return self._extract_literal_any_shape(right, text_fn, root, is_python)

    @staticmethod
    def _is_ancestor(potential_ancestor, node) -> bool:
        """Is `potential_ancestor` an ancestor of `node` in the tree?
        Used to catch self-referential assignments (`url = url + "/x"`) —
        without this check, that assignment would match itself as its own
        "preceding definition" (its LHS position is before the `url`
        reference sitting inside its own RHS), causing unbounded
        recursion: resolving the assignment requires resolving the
        assignment. Excluding an assignment that contains the reference
        within its own right-hand side fixes this at the source, for
        every caller that funnels through here (string wildcards,
        template wildcards, concatenation, direct identifier args).

        Compares via `.id`, not `is` — see _enclosing_function's
        docstring. Using `is` here was the actual reason this fix didn't
        reliably take effect against real code (found via testing against
        swagger-petstore/PetStoreAPI): two references that both land on
        the same tree position via separate `.parent` traversals aren't
        guaranteed to be the same Python object, so `is` silently failed
        to recognize the self-reference in some files while appearing to
        work in others."""
        cur = node.parent
        while cur is not None:
            if cur.id == potential_ancestor.id:
                return True
            cur = cur.parent
        return False

    def _resolve_in_scope(
        self, name: str, reference_node, scope_node, text_fn, is_python: bool,
        skip_nested_functions: bool, true_root,
    ) -> tuple[str | None, bool]:
        """Searches `scope_node`'s subtree for assignments to `name`
        preceding `reference_node`. Returns (resolved_literal_or_None,
        found_any_candidates) — the second value is what lets the caller
        distinguish "no local assignment exists, fall back to globals"
        from "a local assignment exists but is ambiguous, don't fall
        back".

        `scope_node` bounds THIS search (and branch-exclusivity checks
        within it) to the current scope tier. `true_root` is threaded
        through separately to any nested resolution (an f-string whose
        interpolation itself references another variable) so that nested
        call can still correctly fall back all the way to true
        module-level globals — using `scope_node` for that would
        incorrectly cap the nested search at the current tier's boundary."""
        function_types = ("function_definition",) if is_python else (
            "function_declaration", "arrow_function", "function_expression", "method_definition",
        )
        candidates: list[tuple[object, str]] = []

        def collect(node, is_scope_root=False):
            if skip_nested_functions and not is_scope_root and node.type in function_types:
                return  # true-global search: don't dip into other functions' locals
            if is_python and node.type == "assignment":
                left = node.child_by_field_name("left")
                right = node.child_by_field_name("right")
                if (
                    left is not None and right is not None
                    and left.type == "identifier" and text_fn(left) == name
                    and node.start_byte < reference_node.start_byte
                    and not self._is_ancestor(node, reference_node)
                ):
                    literal = self._extract_literal_any_shape(right, text_fn, true_root, is_python)
                    if literal is not None:
                        candidates.append((node, literal))
            elif not is_python and node.type == "variable_declarator":
                name_node = node.child_by_field_name("name")
                value_node = node.child_by_field_name("value")
                if (
                    name_node is not None and value_node is not None
                    and text_fn(name_node) == name
                    and node.start_byte < reference_node.start_byte
                    and not self._is_ancestor(node, reference_node)
                ):
                    literal = self._extract_literal_any_shape(value_node, text_fn, true_root, is_python)
                    if literal is not None:
                        candidates.append((node, literal))
            for child in node.children:
                collect(child)

        collect(scope_node, is_scope_root=True)

        if not candidates:
            return None, False

        compatible = [
            (n, lit) for n, lit in candidates
            if not self._mutually_exclusive(n, reference_node, scope_node)
        ]
        if not compatible:
            return None, True  # local assignment(s) exist but none reach here — still shadows global

        distinct_literals = {lit for _, lit in compatible}
        if len(distinct_literals) == 1:
            return next(iter(distinct_literals)), True

        # Disagreeing values — check whether any pair of them is mutually
        # exclusive with each other. If so, we genuinely can't tell which
        # one applies without knowing which branch ran — refuse.
        for i in range(len(compatible)):
            for j in range(i + 1, len(compatible)):
                node_i, lit_i = compatible[i]
                node_j, lit_j = compatible[j]
                if lit_i != lit_j and self._mutually_exclusive(node_i, node_j, scope_node):
                    return None, True

        # Disagreeing values but not mutually exclusive with each other —
        # straight-line reassignment. Nearest preceding one wins.
        compatible.sort(key=lambda pair: pair[0].start_byte)
        return compatible[-1][1], True

    # --- file discovery ------------------------------------------------

    def _iter_source_files(self):
        """Prefers `git ls-files` (fast — respects .gitignore, no per-directory
        stat storm) and falls back to os.walk for non-git directories or if
        git isn't available."""
        git_files = self._try_git_ls_files()
        if git_files is not None:
            for rel_path in git_files:
                ext = os.path.splitext(rel_path)[1]
                if ext in SUPPORTED_EXTENSIONS:
                    yield os.path.join(self.local_path, rel_path)
            return

        for root, dirs, files in os.walk(self.local_path):
            dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
            for fname in files:
                ext = os.path.splitext(fname)[1]
                if ext in SUPPORTED_EXTENSIONS:
                    yield os.path.join(root, fname)

    def _try_git_ls_files(self) -> list[str] | None:
        try:
            result = subprocess.run(
                ["git", "-C", self.local_path, "ls-files"],
                check=True, capture_output=True, text=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
        return [line for line in result.stdout.splitlines() if line]

    # --- main entrypoint -------------------------------------------------

    def build_graph(self) -> RepoGraph:
        graph = RepoGraph(repo_id=self.repo_id)
        # file_path -> file RepoNode, needed to resolve local imports later
        file_nodes: dict[str, RepoNode] = {}
        skipped_files: list[tuple[str, str]] = []

        for abs_path in self._iter_source_files():
            # Normalize to forward slashes unconditionally — os.path.relpath
            # returns OS-native separators, meaning this was backslash on
            # Windows while every OTHER module that reads file_path (the
            # Semantic Indexer's git-ls-files-derived paths, Cross-Repo
            # Linker's lookups) uses forward slashes unconditionally. That
            # mismatch silently broke cross-module path matching — e.g. the
            # Semantic Indexer's function/class-precise chunking looking up
            # graph_nodes by file_path never matched anything on Windows,
            # falling back to windowed chunking for every file without any
            # visible error. Normalizing once here, at the point rel_path is
            # authoritatively established, fixes it for every downstream
            # consumer at once rather than requiring each one to work around
            # it independently (which is what _resolve_local_imports's own
            # norm_lookup was already doing, locally, before this fix).
            rel_path = os.path.relpath(abs_path, self.local_path).replace(os.sep, "/")
            ext = os.path.splitext(abs_path)[1]
            lang_key = SUPPORTED_EXTENSIONS[ext]

            # Each file's nodes/edges go into a scratch graph first, not the
            # real one directly — _process_python_file/_process_ts_js_file
            # append as they walk, so a mid-file crash (RecursionError or
            # otherwise) would otherwise leave partial function/class nodes
            # and edges behind even though the file "failed". Only merging
            # in on success keeps a skipped file fully absent rather than
            # half-present.
            scratch = RepoGraph(repo_id=self.repo_id)
            try:
                with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
                    source = f.read()
                source_bytes = source.encode("utf-8")

                tree = self._parsers[lang_key].parse(source_bytes)

                file_node = RepoNode(
                    id=str(uuid.uuid4()),
                    node_type=NodeType.FILE,
                    qualified_name=rel_path,
                    file_path=rel_path,
                    metadata={"language": lang_key},
                )
                scratch.nodes.append(file_node)

                if lang_key == "python":
                    self._process_python_file(tree, source_bytes, file_node, scratch)
                else:
                    self._process_ts_js_file(tree, source_bytes, file_node, scratch)

            except RecursionError as e:
                # A pathological self-referential pattern our static
                # analysis can't safely unwind (the known cause — a
                # self-referential assignment — is now excluded upstream
                # in _resolve_in_scope, but this stays as a backstop for
                # any shape we haven't seen yet). One file's parse
                # failing must never abort mapping the rest of the repo.
                skipped_files.append((rel_path, f"RecursionError: {e}"))
                continue
            except Exception as e:
                # Any other per-file failure (malformed source, an
                # unexpected grammar node shape, etc.) — same principle:
                # isolate it to this file, keep going.
                skipped_files.append((rel_path, f"{type(e).__name__}: {e}"))
                continue

            graph.nodes.extend(scratch.nodes)
            graph.edges.extend(scratch.edges)
            file_nodes[rel_path] = scratch.nodes[0]  # the FILE node, always first

        self._resolve_local_imports(graph, file_nodes)

        if skipped_files:
            # Not raised — surfaced as data so the caller (and eventually
            # a staleness/health dashboard) can see coverage gaps instead
            # of them being silent.
            graph.nodes.append(RepoNode(
                id=str(uuid.uuid4()),
                node_type=NodeType.MODULE,
                qualified_name="__mapping_errors__",
                file_path="",
                metadata={"skipped_files": [
                    {"file_path": path, "error": err} for path, err in skipped_files
                ]},
            ))

        return graph

    # --- Python ---------------------------------------------------------

    def _process_python_file(self, tree, source_bytes, file_node, graph):
        root = tree.root_node
        local_defs: dict[str, RepoNode] = {}  # name -> node, for same-file call resolution

        def text(node) -> str:
            return source_bytes[node.start_byte:node.end_byte].decode("utf-8")

        def walk(node, depth=0):
            if node.type == "import_statement" or node.type == "import_from_statement":
                module_text = text(node)
                file_node.metadata.setdefault("raw_imports", []).append(module_text)

            elif node.type == "function_definition" and depth <= 1:
                name_node = node.child_by_field_name("name")
                if name_node:
                    fn_name = text(name_node)
                    fn_node = RepoNode(
                        id=str(uuid.uuid4()),
                        node_type=NodeType.FUNCTION,
                        qualified_name=f"{file_node.qualified_name}::{fn_name}",
                        file_path=file_node.file_path,
                        metadata=self._line_range_metadata(node),
                    )
                    graph.nodes.append(fn_node)
                    local_defs[fn_name] = fn_node
                    graph.edges.append(RepoEdge(
                        id=str(uuid.uuid4()),
                        source_node_id=file_node.id,
                        target_node_id=fn_node.id,
                        edge_type=EdgeType.DEFINES,
                    ))

            elif node.type == "class_definition" and depth <= 1:
                name_node = node.child_by_field_name("name")
                if name_node:
                    cls_name = text(name_node)
                    cls_node = RepoNode(
                        id=str(uuid.uuid4()),
                        node_type=NodeType.CLASS,
                        qualified_name=f"{file_node.qualified_name}::{cls_name}",
                        file_path=file_node.file_path,
                        metadata=self._line_range_metadata(node),
                    )
                    graph.nodes.append(cls_node)
                    local_defs[cls_name] = cls_node
                    graph.edges.append(RepoEdge(
                        id=str(uuid.uuid4()),
                        source_node_id=file_node.id,
                        target_node_id=cls_node.id,
                        edge_type=EdgeType.DEFINES,
                    ))

            for child in node.children:
                walk(child, depth + 1)

        walk(root)

        # Second pass: resolve same-file calls now that local_defs is complete,
        # and capture HTTP call-site literals for the Cross-Repo Linker.
        def walk_calls(node):
            if node.type == "call":
                fn_field = node.child_by_field_name("function")
                if fn_field is not None and fn_field.type == "identifier":
                    callee_name = text(fn_field)
                    if callee_name in local_defs:
                        graph.edges.append(RepoEdge(
                            id=str(uuid.uuid4()),
                            source_node_id=file_node.id,
                            target_node_id=local_defs[callee_name].id,
                            edge_type=EdgeType.CALLS,
                        ))
                http_literal = self._extract_http_call_literal(node, text, root, is_python=True)
                if http_literal is not None:
                    file_node.metadata.setdefault("call_site_literals", []).append(http_literal)
            for child in node.children:
                walk_calls(child)

        walk_calls(root)

    # --- TypeScript / JavaScript -----------------------------------------

    def _process_ts_js_file(self, tree, source_bytes, file_node, graph):
        root = tree.root_node
        local_defs: dict[str, RepoNode] = {}

        def text(node) -> str:
            return source_bytes[node.start_byte:node.end_byte].decode("utf-8")

        def walk(node, depth=0):
            if node.type == "import_statement":
                file_node.metadata.setdefault("raw_imports", []).append(text(node))

            elif node.type == "function_declaration" and depth <= 1:
                name_node = node.child_by_field_name("name")
                if name_node:
                    fn_name = text(name_node)
                    fn_node = RepoNode(
                        id=str(uuid.uuid4()),
                        node_type=NodeType.FUNCTION,
                        qualified_name=f"{file_node.qualified_name}::{fn_name}",
                        file_path=file_node.file_path,
                        metadata=self._line_range_metadata(node),
                    )
                    graph.nodes.append(fn_node)
                    local_defs[fn_name] = fn_node
                    graph.edges.append(RepoEdge(
                        id=str(uuid.uuid4()),
                        source_node_id=file_node.id,
                        target_node_id=fn_node.id,
                        edge_type=EdgeType.DEFINES,
                    ))

            elif node.type == "class_declaration" and depth <= 1:
                name_node = node.child_by_field_name("name")
                if name_node:
                    cls_name = text(name_node)
                    cls_node = RepoNode(
                        id=str(uuid.uuid4()),
                        node_type=NodeType.CLASS,
                        qualified_name=f"{file_node.qualified_name}::{cls_name}",
                        file_path=file_node.file_path,
                        metadata=self._line_range_metadata(node),
                    )
                    graph.nodes.append(cls_node)
                    local_defs[cls_name] = cls_node
                    graph.edges.append(RepoEdge(
                        id=str(uuid.uuid4()),
                        source_node_id=file_node.id,
                        target_node_id=cls_node.id,
                        edge_type=EdgeType.DEFINES,
                    ))

            # const foo = () => {...} / function expressions assigned to a name
            elif node.type == "lexical_declaration" and depth <= 1:
                for declarator in node.children:
                    if declarator.type == "variable_declarator":
                        name_node = declarator.child_by_field_name("name")
                        value_node = declarator.child_by_field_name("value")
                        if (
                            name_node is not None
                            and value_node is not None
                            and value_node.type in ("arrow_function", "function_expression")
                        ):
                            fn_name = text(name_node)
                            fn_node = RepoNode(
                                id=str(uuid.uuid4()),
                                node_type=NodeType.FUNCTION,
                                qualified_name=f"{file_node.qualified_name}::{fn_name}",
                                file_path=file_node.file_path,
                                metadata=self._line_range_metadata(node),
                            )
                            graph.nodes.append(fn_node)
                            local_defs[fn_name] = fn_node
                            graph.edges.append(RepoEdge(
                                id=str(uuid.uuid4()),
                                source_node_id=file_node.id,
                                target_node_id=fn_node.id,
                                edge_type=EdgeType.DEFINES,
                            ))

            for child in node.children:
                walk(child, depth + 1)

        walk(root)

        def walk_calls(node):
            if node.type == "call_expression":
                fn_field = node.child_by_field_name("function")
                if fn_field is not None and fn_field.type == "identifier":
                    callee_name = text(fn_field)
                    if callee_name in local_defs:
                        graph.edges.append(RepoEdge(
                            id=str(uuid.uuid4()),
                            source_node_id=file_node.id,
                            target_node_id=local_defs[callee_name].id,
                            edge_type=EdgeType.CALLS,
                        ))
                http_literal = self._extract_http_call_literal(node, text, root, is_python=False)
                if http_literal is not None:
                    file_node.metadata.setdefault("call_site_literals", []).append(http_literal)
            for child in node.children:
                walk_calls(child)

        walk_calls(root)

    # --- Import resolution -------------------------------------------

    def _resolve_local_imports(self, graph: RepoGraph, file_nodes: dict[str, RepoNode]):
        """
        Best-effort resolution of local imports to file nodes in this repo.
        External package imports are left in each file node's
        `metadata["raw_imports"]` without a graph edge — there's no local
        node for them to point to.

        Tries multiple candidate resolutions per import (relative-dot,
        absolute package-style, and bare same-directory) because real Python
        codebases mix all three depending on how a module is meant to be
        run — e.g. `from .tools import x`, `from agents.tools import x`, and
        `from resilience import x` (bare, resolved via same-directory
        sys.path tricks) can all appear for the same underlying import
        across different files. A bare-name candidate is generated
        regardless of whether it looks local, but it only produces an edge
        if it happens to match an actual file in this repo — so a bare
        import of a genuine external package (`from os import path`) is
        harmless; it just won't match anything in norm_lookup.

        TODO (later hardening pass): proper Python package resolution
        (respecting __init__.py, namespace packages, sys.path manipulation
        beyond same-directory) and TS path-alias resolution (tsconfig
        "paths"). This pass only handles the common cases seen so far.
        """
        norm_lookup = {
            os.path.normpath(path).replace(os.sep, "/"): node
            for path, node in file_nodes.items()
        }

        seen_edges: set[tuple[str, str]] = set()  # dedupe e.g. try/except dual-style imports
        for file_node in list(file_nodes.values()):
            raw_imports = file_node.metadata.get("raw_imports", [])
            for raw in raw_imports:
                candidates = self._guess_local_import_candidates(file_node.file_path, raw)
                for candidate in candidates:
                    target = norm_lookup.get(candidate)
                    if target is None:
                        continue
                    edge_key = (file_node.id, target.id)
                    if edge_key in seen_edges:
                        break
                    seen_edges.add(edge_key)
                    graph.edges.append(RepoEdge(
                        id=str(uuid.uuid4()),
                        source_node_id=file_node.id,
                        target_node_id=target.id,
                        edge_type=EdgeType.IMPORTS,
                    ))
                    break  # first matching candidate wins, don't also try the rest

    @staticmethod
    def _guess_local_import_candidates(source_file_rel: str, raw_import: str) -> list[str]:
        """Returns normalized relative-path guesses to try, in priority
        order. Empty list if this clearly isn't a Python/TS-style import
        we handle."""
        source_dir = os.path.dirname(source_file_rel)
        candidates: list[str] = []

        # Python: "from .foo import bar" / "from ..pkg.foo import bar"
        if raw_import.startswith("from ."):
            after_from = raw_import[len("from "):].split(" import")[0].strip()
            dots = len(after_from) - len(after_from.lstrip("."))
            module_part = after_from.lstrip(".").replace(".", "/")
            base = source_dir
            for _ in range(dots - 1):
                base = os.path.dirname(base)
            guess = os.path.normpath(os.path.join(base, module_part) + ".py") if module_part else \
                os.path.normpath(os.path.join(base, "__init__.py"))
            candidates.append(guess.replace(os.sep, "/"))
            return candidates

        # TS/JS: import ... from './foo' or '../foo'
        if "from '" in raw_import or 'from "' in raw_import:
            quote = "'" if "from '" in raw_import else '"'
            spec = raw_import.split(f"from {quote}")[1].split(quote)[0]
            if spec.startswith("."):
                base = os.path.normpath(os.path.join(source_dir, spec))
                # try common extensions; caller does exact-match lookup, so
                # list a few in priority order
                for ext in (".ts", ".tsx", ".js", ".jsx"):
                    candidates.append((base + ext).replace(os.sep, "/"))
            return candidates  # bare specifier (no leading '.') -> external package

        # Python: "from agents.tools import x" (absolute package-style) or
        # "from resilience import x" (bare — same-directory in practice)
        if raw_import.startswith("from "):
            module_part = raw_import[len("from "):].split(" import")[0].strip()
            parts = module_part.split(".")
            # Absolute, package-rooted: agents/tools.py from repo root
            candidates.append("/".join(parts) + ".py")
            candidates.append("/".join(parts) + "/__init__.py")
            # Bare same-directory: resilience.py next to the importing file
            if len(parts) == 1:
                same_dir = os.path.normpath(os.path.join(source_dir, parts[0] + ".py"))
                candidates.append(same_dir.replace(os.sep, "/"))
            return candidates

        # Python: "import agents.tools" / "import resilience"
        if raw_import.startswith("import "):
            module_part = raw_import[len("import "):].split(" as ")[0].strip()
            parts = module_part.split(".")
            candidates.append("/".join(parts) + ".py")
            if len(parts) == 1:
                same_dir = os.path.normpath(os.path.join(source_dir, parts[0] + ".py"))
                candidates.append(same_dir.replace(os.sep, "/"))
            return candidates

        return candidates


def persist_graph(graph: RepoGraph) -> None:
    """Writes a built RepoGraph to graph_nodes / graph_edges."""
    with connection() as conn:
        with conn.cursor() as cur:
            for node in graph.nodes:
                cur.execute(
                    """
                    INSERT INTO graph_nodes
                        (id, repo_id, node_type, qualified_name, file_path, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        node.id, graph.repo_id, node.node_type.value,
                        node.qualified_name, node.file_path,
                        json.dumps(node.metadata),
                    ),
                )
            for edge in graph.edges:
                cur.execute(
                    """
                    INSERT INTO graph_edges
                        (id, source_node_id, target_node_id, edge_type, is_cross_repo, confidence)
                    VALUES (%s, %s, %s, %s, false, 'confirmed')
                    """,
                    (edge.id, edge.source_node_id, edge.target_node_id, edge.edge_type.value),
                )