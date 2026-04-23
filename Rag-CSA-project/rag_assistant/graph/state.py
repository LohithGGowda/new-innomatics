"""
Graph State Schema
------------------
The single TypedDict that flows through every node in the LangGraph StateGraph.

Design principle: all data produced by any node is written into this state object.
Nodes read only what they need and write only what they produce — no side channels.
"""

from __future__ import annotations

from typing import List, Optional, TypedDict

from rag_assistant.ingestion.vector_store import RetrievedChunk


class GraphState(TypedDict, total=False):
    """
    Shared state object passed between LangGraph nodes.

    Fields
    ------
    query : str
        The raw user question.

    retrieved_chunks : List[RetrievedChunk]
        Top-K chunks returned by the vector store.

    top_similarity : float
        Highest cosine similarity score among retrieved chunks.
        Used by the Router to decide escalation.

    synthesized_context : str
        Concatenated chunk texts formatted as a context block for the LLM prompt.

    intent : str
        Detected intent label: "answer" | "escalate".
        Set by the Router node.

    escalation_reason : Optional[str]
        Human-readable reason for escalation (low confidence, missing context, etc.).

    llm_response : Optional[str]
        Final answer produced by the Generator node.

    human_response : Optional[str]
        Answer provided by a human agent during HITL escalation.

    final_answer : str
        The answer surfaced to the user — either llm_response or human_response.

    error : Optional[str]
        Any error message that caused the workflow to short-circuit.

    node_trace : List[str]
        Ordered list of node names visited — useful for debugging and audit logs.
    """

    query: str
    retrieved_chunks: List[RetrievedChunk]
    top_similarity: float
    synthesized_context: str
    intent: str                    # "answer" | "escalate"
    escalation_reason: Optional[str]
    llm_response: Optional[str]
    human_response: Optional[str]
    final_answer: str
    error: Optional[str]
    node_trace: List[str]
