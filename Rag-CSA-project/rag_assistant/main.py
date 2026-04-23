"""
RAG Customer Support Assistant — Entry Point
--------------------------------------------
CLI interface that ties together ingestion, graph execution, and HITL.

Usage:
  # Ingest a PDF knowledge base
  python -m rag_assistant.main ingest --pdf path/to/manual.pdf

  # Start the interactive support chat
  python -m rag_assistant.main chat

  # Single query (non-interactive)
  python -m rag_assistant.main query --text "How do I reset my password?"

  # Reset the vector store (destructive!)
  python -m rag_assistant.main reset
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

from rag_assistant.config import config
from rag_assistant.graph.state import GraphState
from rag_assistant.graph.workflow import build_graph
from rag_assistant.hitl.handler import handle_escalation
from rag_assistant.ingestion.pipeline import get_vector_store, ingest_pdfs

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, config.log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ANSI colours
_BOLD  = "\033[1m"
_GREEN = "\033[92m"
_CYAN  = "\033[96m"
_RESET = "\033[0m"


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_query(query: str, graph) -> str:
    """
    Execute the LangGraph workflow for a single query and return the final answer.

    Handles HITL escalation inline by calling the CLI handler when the graph
    routes to the hitl node.

    Args:
        query: User's question string.
        graph: Compiled LangGraph application.

    Returns:
        Final answer string (from LLM or human agent).
    """
    initial_state: GraphState = {
        "query": query,
        "node_trace": [],
    }

    logger.info("Running graph for query: %s", query)
    final_state: GraphState = graph.invoke(initial_state)

    trace = final_state.get("node_trace", [])
    logger.debug("Node trace: %s", " → ".join(trace))

    # Check if HITL was triggered
    if "hitl" in trace:
        human_response = handle_escalation(
            query=query,
            escalation_reason=final_state.get("escalation_reason", "Unknown reason"),
            llm_draft=final_state.get("llm_response"),
        )
        final_state["human_response"] = human_response
        final_state["final_answer"] = human_response

    answer = final_state.get("final_answer") or final_state.get("llm_response")

    if not answer:
        error = final_state.get("error", "Unknown error")
        answer = f"[System Error] {error}"

    return answer


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def cmd_ingest(args: argparse.Namespace) -> None:
    """Ingest one or more PDF files into ChromaDB."""
    pdf_paths: List[Path] = [Path(p) for p in args.pdf]
    missing = [p for p in pdf_paths if not p.exists()]
    if missing:
        print(f"[ERROR] PDF(s) not found: {', '.join(str(p) for p in missing)}")
        sys.exit(1)

    print(f"Ingesting {len(pdf_paths)} PDF(s)…")
    store = ingest_pdfs(pdf_paths)
    print(f"{_GREEN}✓ Ingestion complete. {store.count()} chunks stored.{_RESET}")


def cmd_chat(args: argparse.Namespace) -> None:
    """Start an interactive chat session."""
    print(f"\n{_BOLD}RAG Customer Support Assistant{_RESET}")
    print(f"Embedding provider : {config.embedding_provider}")
    print(f"Similarity threshold: {config.similarity_threshold}")
    print("Type 'exit' or 'quit' to end the session.\n")

    store = get_vector_store()
    if store.count() == 0:
        print(
            "[WARNING] The knowledge base is empty. "
            "Run 'ingest' first to load a PDF.\n"
        )

    graph = build_graph(store)

    while True:
        try:
            user_input = input(f"{_CYAN}You:{_RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        answer = run_query(user_input, graph)
        print(f"\n{_GREEN}Assistant:{_RESET} {answer}\n")


def cmd_query(args: argparse.Namespace) -> None:
    """Run a single non-interactive query."""
    store = get_vector_store()
    graph = build_graph(store)
    answer = run_query(args.text, graph)
    print(f"\n{_GREEN}Answer:{_RESET} {answer}\n")


def cmd_reset(args: argparse.Namespace) -> None:
    """Reset (wipe) the ChromaDB collection."""
    confirm = input(
        "[WARNING] This will delete all stored embeddings. Type 'yes' to confirm: "
    ).strip()
    if confirm.lower() != "yes":
        print("Aborted.")
        return
    store = get_vector_store()
    store.reset()
    print(f"{_GREEN}✓ Vector store reset.{_RESET}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag_assistant",
        description="RAG-based Customer Support Assistant with LangGraph & HITL",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest PDF(s) into ChromaDB")
    ingest_parser.add_argument(
        "--pdf",
        nargs="+",
        required=True,
        metavar="PATH",
        help="Path(s) to PDF file(s) to ingest",
    )
    ingest_parser.set_defaults(func=cmd_ingest)

    # chat
    chat_parser = subparsers.add_parser("chat", help="Start interactive chat session")
    chat_parser.set_defaults(func=cmd_chat)

    # query
    query_parser = subparsers.add_parser("query", help="Run a single query")
    query_parser.add_argument("--text", required=True, help="Query text")
    query_parser.set_defaults(func=cmd_query)

    # reset
    reset_parser = subparsers.add_parser("reset", help="Reset the vector store")
    reset_parser.set_defaults(func=cmd_reset)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
