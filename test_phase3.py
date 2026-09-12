"""
Phase 3 smoke test.

Wires Router -> Retrieval Agent -> Answer Synthesizer as a plain
sequential pipeline (not LangGraph — that's Phase 8's job; this just
validates the three agents work correctly together first).

Usage:
    python test_phase3.py "<question>" [user_id]

Example:
    python test_phase3.py "how does the n8n webhook retry logic work"
"""
import sys

from agents.router import Router
from agents.retrieval_agent import RetrievalAgent
from agents.answer_synthesizer import AnswerSynthesizer


def main():
    if len(sys.argv) < 2:
        print('Usage: python test_phase3.py "<question>" [user_id]')
        sys.exit(1)

    question = sys.argv[1]
    user_id = sys.argv[2] if len(sys.argv) > 2 else "test-user"

    router = Router()
    retrieval = RetrievalAgent()
    synthesizer = AnswerSynthesizer()

    print(f"Question: {question!r}\n")

    print("--- Router ---")
    route_result = router.route(question, user_id)
    print(f"question_type: {route_result['question_type']}")
    print(f"repo_ids: {route_result['repo_ids']}")
    if route_result["needs_clarification"]:
        print(f"[needs clarification]: {route_result['clarification_message']}")
        print("(proceeding with best-effort scope anyway — no human in the loop in this test script)")

    print("\n--- Retrieval Agent ---")
    chunks = retrieval.retrieve(question, route_result["repo_ids"], user_id)
    print(f"{len(chunks)} chunks retrieved")
    for c in chunks[:5]:
        location = c.get("file_path") or f"({c.get('source_type')})"
        print(f"  [{c['source_type']}] {location} (similarity={c['similarity']:.3f})")

    print("\n--- Answer Synthesizer ---\n")
    result = synthesizer.synthesize(question, chunks)
    print(result["answer"])


if __name__ == "__main__":
    main()