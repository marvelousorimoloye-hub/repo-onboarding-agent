"""
Phase 4 smoke test.

Finds a starting graph node by (partial) qualified name within a repo,
runs a multi-hop trace from there, prints each hop, then feeds the trace
into AnswerSynthesizer (Phase 3, reused as-is via trace_to_chunks) for a
final grounded answer.

Usage:
    python test_phase4.py <repo_name> "<qualified_name_substring>" "<question>" [max_hops]

Example:
    python test_phase4.py wax-co-agentic-infrastructure "call_n8n_webhook" \\
        "how does the webhook call relate to the circuit breaker" 4
"""
import sys

from ingestion.repo_registry import RepoRegistry
from db.connection import connection
from agents.trace_agent import TraceAgent
from agents.answer_synthesizer import AnswerSynthesizer


def find_starting_node(repo_id: str, name_substring: str):
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, qualified_name FROM graph_nodes
                WHERE repo_id = %s AND qualified_name ILIKE %s
                LIMIT 5
                """,
                (repo_id, f"%{name_substring}%"),
            )
            return cur.fetchall()


def main():
    if len(sys.argv) < 4:
        print('Usage: python test_phase4.py <repo_name> "<qualified_name_substring>" "<question>" [max_hops]')
        sys.exit(1)

    repo_name, name_substring, question = sys.argv[1], sys.argv[2], sys.argv[3]
    max_hops = int(sys.argv[4]) if len(sys.argv) > 4 else 5

    registry = RepoRegistry()
    record = registry.get_repo_by_name(repo_name)
    if record is None:
        print(f"No repo registered with name '{repo_name}'.")
        sys.exit(1)

    matches = find_starting_node(record.id, name_substring)
    if not matches:
        print(f"No graph node found matching '{name_substring}' in {repo_name}.")
        sys.exit(1)
    if len(matches) > 1:
        print(f"Multiple matches, using the first: {[m[1] for m in matches]}")
    starting_node_id, starting_qualified_name = str(matches[0][0]), matches[0][1]
    print(f"Starting from: {starting_qualified_name}\n")

    tracer = TraceAgent()
    result = tracer.trace(starting_node_id, question, max_hops=max_hops)

    print(f"--- Trace ({result['hops_taken']} hops, stopped: {result['stopped_reason']}) ---")
    for step in result["trace"]:
        print(f"  hop {step['hop']}: {step['qualified_name']} ({step['file_path']})")

    print("\n--- Answer Synthesizer (from trace) ---\n")
    synthesizer = AnswerSynthesizer()
    chunks = tracer.trace_to_chunks(result)
    answer_result = synthesizer.synthesize(question, chunks)
    print(answer_result["answer"])


if __name__ == "__main__":
    main()