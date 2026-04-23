"""
LangGraph Workflow Assembly
---------------------------
Builds and compiles the StateGraph that wires together:

  retriever_node → router_node → [generator_node | hitl_node]

Conditional edge from router_node:
  - intent == "answer"   → generator_node
  - intent == "escalate" → hitl_node

The compiled graph is returned as a callable that accepts a GraphState dict.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Literal

from langgraph.graph import END, StateGraph

from rag_assistant.graph.nodes import (
    generator_node,
    hitl_node,
    retriever_node,
    router_node,
)
from rag_assistant.graph.state import GraphState
from rag_assistant.ingestion.vector_store import VectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Routing function (used as conditional edge)
# ---------------------------------------------------------------------------

def _route_after_router(state: GraphState) -> Literal["generator", "hitl"]:
    """
    Determine the next node after the router.

    Returns "generator" if intent is "answer", otherwise "hitl".
    """
    intent = state.get("intent", "escalate")
    if intent == "answer":
        return "generator"
    return "hitl"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(vector_store: VectorStore) -> StateGraph:
    """
    Construct and compile the RAG workflow graph.

    Args:
        vector_store: Initialised VectorStore used by the retriever node.

    Returns:
        A compiled LangGraph application.
    """
    # Bind the vector_store into the retriever node via partial application
    bound_retriever = partial(retriever_node, vector_store=vector_store)

    graph = StateGraph(GraphState)

    # Register nodes
    graph.add_node("retriever", bound_retriever)
    graph.add_node("router", router_node)
    graph.add_node("generator", generator_node)
    graph.add_node("hitl", hitl_node)

    # Entry point
    graph.set_entry_point("retriever")

    # Fixed edges
    graph.add_edge("retriever", "router")

    # Conditional edge: router → generator OR hitl
    graph.add_conditional_edges(
        "router",
        _route_after_router,
        {
            "generator": "generator",
            "hitl": "hitl",
        },
    )

    # Terminal edges
    graph.add_edge("generator", END)
    graph.add_edge("hitl", END)

    compiled = graph.compile()
    logger.info("LangGraph workflow compiled successfully.")
    return compiled
