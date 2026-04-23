"""
LangGraph Nodes
---------------
Each function here is a LangGraph node: it receives the current GraphState,
performs its work, and returns a dict of state updates.

Node A — retriever_node   : Query ChromaDB, synthesize context
Node B — router_node      : Evaluate confidence, set intent
Node C — generator_node   : Call LLM, produce final answer
Node D — hitl_node        : Pause for human input (HITL escalation)
"""

from __future__ import annotations

import logging
import textwrap
from typing import Dict, Any

from rag_assistant.config import config
from rag_assistant.graph.state import GraphState
from rag_assistant.ingestion.vector_store import VectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _append_trace(state: GraphState, node_name: str) -> list:
    trace = list(state.get("node_trace") or [])
    trace.append(node_name)
    return trace


def _build_context(chunks) -> str:
    """Format retrieved chunks into a numbered context block."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[{i}] (source: {chunk.source.split('/')[-1]}, "
            f"page {chunk.page_number}, similarity {chunk.similarity:.2f})\n"
            f"{chunk.text}"
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Node A: Retriever
# ---------------------------------------------------------------------------

def retriever_node(state: GraphState, vector_store: VectorStore) -> Dict[str, Any]:
    """
    Query the vector store and synthesize a context block.

    Reads  : state["query"]
    Writes : retrieved_chunks, top_similarity, synthesized_context
    """
    node_name = "retriever"
    logger.info("[%s] Retrieving for query: %s", node_name, state["query"])

    try:
        chunks = vector_store.query(
            query_text=state["query"],
            top_k=config.top_k_results,
        )
    except Exception as exc:
        logger.error("[%s] Vector store query failed: %s", node_name, exc)
        return {
            "error": f"Retrieval failed: {exc}",
            "node_trace": _append_trace(state, node_name),
        }

    if not chunks:
        return {
            "retrieved_chunks": [],
            "top_similarity": 0.0,
            "synthesized_context": "",
            "node_trace": _append_trace(state, node_name),
        }

    top_sim = chunks[0].similarity
    context = _build_context(chunks)

    logger.info(
        "[%s] Retrieved %d chunks; top_similarity=%.3f",
        node_name, len(chunks), top_sim,
    )

    return {
        "retrieved_chunks": chunks,
        "top_similarity": top_sim,
        "synthesized_context": context,
        "node_trace": _append_trace(state, node_name),
    }


# ---------------------------------------------------------------------------
# Node B: Router
# ---------------------------------------------------------------------------

def router_node(state: GraphState) -> Dict[str, Any]:
    """
    Evaluate retrieval confidence and set routing intent.

    Three-tier routing:
    ┌─────────────────────────────────────────────────────────────┐
    │ similarity >= threshold (0.70)  → ANSWER  (generator)       │
    │ similarity in [0.30, 0.70)      → ESCALATE (human agent)    │
    │ similarity < 0.30               → OUT_OF_SCOPE (bot says    │
    │                                   "I don't know" directly)  │
    └─────────────────────────────────────────────────────────────┘

    Rationale:
    - Very low similarity (< 0.30) means the query has nothing to do
      with the knowledge base (e.g. "what's my name", "capital of France").
      Escalating these to a human agent is pointless — the bot should
      politely say it can't help with that topic.
    - Mid-range similarity (0.30–0.70) means the topic is related but
      the system isn't confident enough — a real support question that
      deserves human attention.
    - High similarity (>= 0.70) means the system can answer directly.

    Reads  : top_similarity, retrieved_chunks, error
    Writes : intent, escalation_reason, final_answer (for out-of-scope)
    """
    node_name = "router"
    logger.info("[%s] Evaluating routing intent.", node_name)

    # Hard error from upstream
    if state.get("error"):
        return {
            "intent": "escalate",
            "escalation_reason": f"Upstream error: {state['error']}",
            "node_trace": _append_trace(state, node_name),
        }

    chunks  = state.get("retrieved_chunks") or []
    top_sim = state.get("top_similarity", 0.0)

    # ── Tier 3: Completely out of scope ──────────────────────────────────
    OUT_OF_SCOPE_THRESHOLD = 0.30
    if not chunks or top_sim < OUT_OF_SCOPE_THRESHOLD:
        msg = (
            "I'm sorry, I don't have information about that in my knowledge base. "
            "I can only help with questions related to our products and services — "
            "such as returns, shipping, passwords, or warranties. "
            "Is there anything like that I can help you with?"
        )
        logger.info(
            "[%s] Out-of-scope query (similarity=%.3f) — responding directly.",
            node_name, top_sim,
        )
        return {
            "intent": "out_of_scope",
            "final_answer": msg,
            "escalation_reason": None,
            "node_trace": _append_trace(state, node_name),
        }

    # ── Tier 2: In-scope but low confidence → escalate to human ─────────
    if top_sim < config.similarity_threshold:
        reason = (
            f"Low retrieval confidence (similarity={top_sim:.3f} < "
            f"threshold={config.similarity_threshold}). "
            f"Query is related to the KB but needs human review."
        )
        logger.info("[%s] Escalating to human — %s", node_name, reason)
        return {
            "intent": "escalate",
            "escalation_reason": reason,
            "node_trace": _append_trace(state, node_name),
        }

    # ── Tier 1: High confidence → generate answer ────────────────────────
    logger.info("[%s] Routing to generator (similarity=%.3f).", node_name, top_sim)
    return {
        "intent": "answer",
        "escalation_reason": None,
        "node_trace": _append_trace(state, node_name),
    }


# ---------------------------------------------------------------------------
# Node C: Generator
# ---------------------------------------------------------------------------

def generator_node(state: GraphState) -> Dict[str, Any]:
    """
    Call the LLM with the retrieved context and produce a final answer.

    Reads  : query, synthesized_context
    Writes : llm_response, final_answer
    """
    node_name = "generator"
    logger.info("[%s] Generating answer.", node_name)

    query = state["query"]
    context = state.get("synthesized_context", "")

    prompt = textwrap.dedent(f"""
        You are a helpful customer support assistant.
        Answer the user's question using ONLY the context provided below.
        If the context does not contain enough information to answer confidently,
        say "I'm not sure based on the available information" and suggest
        contacting a human agent.

        Context:
        {context}

        Question: {query}

        Answer:
    """).strip()

    try:
        llm_response = _call_llm(prompt)
    except Exception as exc:
        logger.error("[%s] LLM call failed: %s", node_name, exc)
        return {
            "error": f"LLM generation failed: {exc}",
            "intent": "escalate",
            "escalation_reason": f"LLM failure: {exc}",
            "node_trace": _append_trace(state, node_name),
        }

    # If the LLM itself signals uncertainty, escalate
    uncertainty_phrases = [
        "i'm not sure",
        "i don't know",
        "cannot answer",
        "not enough information",
        "contact a human",
    ]
    if any(phrase in llm_response.lower() for phrase in uncertainty_phrases):
        logger.info("[%s] LLM flagged uncertainty — escalating.", node_name)
        return {
            "llm_response": llm_response,
            "intent": "escalate",
            "escalation_reason": "LLM expressed uncertainty in its response.",
            "node_trace": _append_trace(state, node_name),
        }

    logger.info("[%s] Answer generated successfully.", node_name)
    return {
        "llm_response": llm_response,
        "final_answer": llm_response,
        "node_trace": _append_trace(state, node_name),
    }


def _call_llm(prompt: str) -> str:
    """
    Dispatch to the configured LLM backend.

    OpenAI is used when EMBEDDING_PROVIDER=openai (same API key).
    Falls back to a local HuggingFace pipeline otherwise.
    """
    if config.embedding_provider == "openai" and config.openai_api_key:
        return _call_openai(prompt)
    else:
        return _call_hf_pipeline(prompt)


def _call_openai(prompt: str) -> str:
    from openai import OpenAI  # type: ignore

    client = OpenAI(api_key=config.openai_api_key)
    response = client.chat.completions.create(
        model=config.openai_chat_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


def _call_hf_pipeline(prompt: str) -> str:
    """
    Offline extractive answer generation using sentence-transformers.

    No additional model downloads needed — reuses the all-MiniLM-L6-v2
    model already installed for embeddings.

    Steps:
    1. Parse the question and context out of the structured prompt.
    2. Split context into sentences.
    3. Score each sentence against the question using cosine similarity.
    4. Return the top-3 most relevant sentences as the answer.

    This is fully compatible with any version of transformers/sentence-transformers
    and requires zero API keys.
    """
    import re
    import numpy as np
    from sentence_transformers import SentenceTransformer, util  # type: ignore

    # ── Parse question and context from the structured prompt ────────────
    question = ""
    context_lines = []
    in_context = False

    for line in prompt.splitlines():
        stripped = line.strip()
        if stripped.startswith("Question:"):
            question = stripped.replace("Question:", "").strip()
            in_context = False
        elif stripped == "Context:":
            in_context = True
        elif stripped in ("Answer:", ""):
            continue
        elif in_context and stripped:
            # Skip the [N] (source: page X, similarity Y) header lines
            if not re.match(r"^\[\d+\]", stripped):
                context_lines.append(stripped)

    context = " ".join(context_lines).strip()

    if not question or not context:
        return "I'm not sure based on the available information. Please contact a human agent."

    # ── Split context into sentences ──────────────────────────────────────
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", context)
        if len(s.strip()) > 15
    ]

    if not sentences:
        return "I'm not sure based on the available information. Please contact a human agent."

    # ── Score sentences against the question ─────────────────────────────
    # Reuse the cached embedding model (already loaded for retrieval)
    if not hasattr(_call_hf_pipeline, "_model"):
        logger.info("Loading sentence-transformers model for extractive QA…")
        _call_hf_pipeline._model = SentenceTransformer(  # type: ignore[attr-defined]
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    model = _call_hf_pipeline._model  # type: ignore[attr-defined]
    q_emb  = model.encode(question,  convert_to_tensor=True)
    s_embs = model.encode(sentences, convert_to_tensor=True)
    scores = util.cos_sim(q_emb, s_embs)[0].cpu().numpy()

    # Pick top-3 most relevant sentences, preserve reading order
    top_n   = min(3, len(sentences))
    top_idx = sorted(np.argsort(scores)[-top_n:].tolist())
    best    = [sentences[i] for i in top_idx if scores[i] > 0.25]

    if not best:
        return "I'm not sure based on the available information. Please contact a human agent."

    logger.debug("Extractive answer from %d sentences, top score=%.3f", len(sentences), float(scores.max()))
    return " ".join(best)


# ---------------------------------------------------------------------------
# Node D: HITL
# ---------------------------------------------------------------------------

def hitl_node(state: GraphState) -> Dict[str, Any]:
    """
    Human-in-the-Loop escalation node.

    This node is a LangGraph interrupt point. In the CLI runner it blocks
    and waits for a human agent to type a response. The human's answer is
    stored in human_response and surfaced as final_answer.

    Reads  : query, escalation_reason
    Writes : human_response, final_answer
    """
    node_name = "hitl"
    logger.info("[%s] Escalating to human agent.", node_name)

    # The actual blocking I/O is handled by the HITL handler in hitl/handler.py.
    # This node just marks the state — the graph runner calls the handler.
    return {
        "node_trace": _append_trace(state, node_name),
        # human_response and final_answer are filled in by the HITL handler
        # after the interrupt is resolved.
    }
