# Technical Documentation
## RAG-Based Customer Support Assistant

---

## 1. Introduction

### What is RAG?
Retrieval-Augmented Generation (RAG) is an architecture that combines a retrieval system with a generative language model. Instead of relying solely on the LLM's parametric memory (which can hallucinate or become stale), RAG first retrieves relevant passages from a curated knowledge base and injects them into the LLM's prompt as grounding context.

The result: answers that are factually anchored to your documents, traceable to a source, and updatable without retraining the model.

### Why is it Needed?
| Problem | RAG Solution |
|---|---|
| LLMs hallucinate facts | Answers are grounded in retrieved document chunks |
| Knowledge cutoff dates | Knowledge base is updated by re-ingesting new PDFs |
| No source attribution | Every answer traces back to a specific page and chunk |
| High fine-tuning cost | No model training required — just update the vector store |

### Use Case
A customer support assistant that answers questions about products, policies, and procedures by querying a PDF knowledge base. Queries that fall outside the knowledge base or have low retrieval confidence are escalated to a human agent.

---

## 2. System Architecture Explanation

### Ingestion Pipeline
```
PDF → PDFPlumber → RawDocument (per page)
    → Chunker (512-char windows, 64-char overlap) → Chunk objects
    → Embedding Model → dense vectors
    → ChromaDB (cosine-indexed persistent store)
```

PDFPlumber is chosen over PyPDF2 because it handles complex layouts (tables, multi-column text) more reliably. Each page becomes a `RawDocument`; pages with no extractable text are skipped with a warning.

Chunking uses a character-level sliding window. The 64-character overlap ensures that sentences split across chunk boundaries are still retrievable from either chunk.

### Query Pipeline
```
User query → Embedding Model → query vector
           → ChromaDB cosine search → top-5 RetrievedChunks
           → Retriever Node (context synthesis)
           → Router Node (confidence check)
           → Generator Node (LLM) OR HITL Node (human)
           → final_answer → User
```

### Component Interactions
- `VectorStore` is instantiated once and injected into `retriever_node` via `functools.partial`.
- `GraphState` is the single shared data structure — nodes communicate exclusively through it.
- The HITL handler is called by the runner *after* graph execution, not inside the graph, to keep the graph pure and testable.

---

## 3. Design Decisions

### Chunk Size: 512 characters
- **Too small (< 200 chars):** Chunks lose sentence context; retrieval becomes noisy.
- **Too large (> 1000 chars):** Chunks dilute the embedding signal; similarity scores become less discriminative.
- **512 chars** is a pragmatic middle ground that fits ~3–5 sentences — enough context for most support queries.

### Overlap: 64 characters
- Prevents information loss at chunk boundaries.
- 64 chars ≈ one sentence fragment — enough to preserve cross-boundary context without doubling storage.

### Embedding Strategy
- **HuggingFace all-MiniLM-L6-v2** (default): 384-dimensional vectors, runs on CPU, no API cost. Suitable for offline deployments and development.
- **OpenAI text-embedding-3-small**: Higher quality, 1536-dimensional vectors. Use in production when API access is available.
- The `EmbeddingModel` ABC ensures the rest of the system is backend-agnostic.

### Retrieval Approach
- Cosine similarity over HNSW index (ChromaDB default).
- Top-5 results provide enough context without overwhelming the LLM prompt.
- Similarity threshold of 0.70 is a conservative default — tune upward (0.75–0.80) for stricter quality, downward (0.60) for broader recall.

### Prompt Design
```
You are a helpful customer support assistant.
Answer the user's question using ONLY the context provided below.
If the context does not contain enough information to answer confidently,
say "I'm not sure based on the available information" and suggest
contacting a human agent.

Context:
[numbered chunks with source and page attribution]

Question: <user query>

Answer:
```

Key choices:
- "ONLY the context" — prevents the LLM from mixing in parametric knowledge.
- Explicit uncertainty instruction — triggers the generator's re-escalation logic.
- Numbered chunks with source attribution — enables the LLM to reference specific pages.

---

## 4. Workflow Explanation

### LangGraph Usage
LangGraph's `StateGraph` is used instead of a linear `LangChain` chain because:
1. **Conditional routing** requires branching logic that chains cannot express cleanly.
2. **HITL interrupts** require the ability to pause execution at a specific node.
3. **Auditability** — the `node_trace` field records every node visited, enabling replay and debugging.

### Node Responsibilities

**Node A — Retriever**
Queries ChromaDB with the embedded user query. Synthesises a numbered context block from the top-K results. Records `top_similarity` for the router.

**Node B — Router**
Stateless decision node. Reads `top_similarity` and `retrieved_chunks`. Sets `intent = "answer"` if confidence is sufficient, `"escalate"` otherwise. No I/O side effects.

**Node C — Generator**
Constructs the grounded prompt, calls the LLM, and checks the response for uncertainty phrases. If uncertainty is detected, overrides `intent` to `"escalate"` so the runner triggers HITL.

**Node D — HITL**
Marks the state. The actual blocking I/O happens in `hitl/handler.py` after the graph returns, keeping the graph nodes pure functions.

### State Transitions
```
{query} → retriever → {+chunks, +similarity, +context}
        → router    → {+intent, +escalation_reason}
        → generator → {+llm_response, +final_answer}   [happy path]
        → hitl      → runner calls handler              [escalation path]
                    → {+human_response, +final_answer}
```

---

## 5. Conditional Logic

### Intent Detection
The router uses a simple threshold rule rather than a classifier:
- `top_similarity ≥ 0.70` → `intent = "answer"`
- `top_similarity < 0.70` → `intent = "escalate"`

This is intentionally simple and auditable. A classifier-based approach would add latency and a training data dependency.

### Routing Decisions
```
router_node:
  if error → escalate
  if no chunks → escalate (missing context)
  if top_similarity < threshold → escalate (low confidence)
  else → answer

generator_node (secondary check):
  if LLM response contains uncertainty phrases → escalate
```

The two-layer check (router + generator) catches cases where retrieval scores look acceptable but the LLM still cannot form a confident answer.

---

## 6. HITL Implementation

### Role of Human Intervention
When the system cannot answer with sufficient confidence, a human agent receives:
- The original user query
- The reason for escalation (low similarity score, missing context, LLM uncertainty)
- An optional LLM-generated draft as a starting point

The human types a response directly in the CLI. In a production deployment, this would be replaced by a ticketing system webhook (e.g., Zendesk, Freshdesk) or a Slack notification.

### Benefits
- Prevents incorrect automated answers from reaching customers.
- Provides a feedback signal — human responses can be logged and used to improve the knowledge base.
- Maintains customer trust by acknowledging the system's limitations.

### Limitations
- CLI-based HITL is synchronous — the user waits for the human response.
- No persistence of escalated queries across restarts (in-memory only).
- Human response quality is not validated.

---

## 7. Challenges & Trade-offs

### Retrieval Accuracy vs Speed
- Higher `top_k` → more context → better answers, but larger prompts and higher LLM cost.
- Lower `top_k` → faster, cheaper, but may miss relevant chunks.
- **Decision:** `top_k = 5` balances quality and cost for typical support queries.

### Chunk Size vs Context Quality
- Smaller chunks → more precise retrieval, but fragments lose sentence context.
- Larger chunks → richer context, but embedding signal is diluted.
- **Decision:** 512 chars with 64-char overlap. Revisit if retrieval quality is poor on long-form answers.

### Cost vs Performance
| Mode | Embedding Cost | LLM Cost | Latency |
|---|---|---|---|
| Full offline (HF) | Free | Free | 2–8s (CPU) |
| Hybrid (HF embed + OpenAI LLM) | Free | ~$0.001/query | 1–3s |
| Full OpenAI | ~$0.0001/query | ~$0.001/query | 0.5–2s |

### Similarity Threshold Tuning
- Too high (0.85+): Many valid queries escalate unnecessarily.
- Too low (0.50–): Low-quality answers reach users.
- **Default 0.70** is a starting point. Tune using the testing strategy below.

---

## 8. Testing Strategy

### Unit Tests
Each module is independently testable:
```python
# Test chunker
chunks = chunk_documents([RawDocument(source="test.pdf", page_number=1, text="...")])
assert all(len(c.text) <= 512 for c in chunks)

# Test router escalation
state = {"top_similarity": 0.5, "retrieved_chunks": [...]}
result = router_node(state)
assert result["intent"] == "escalate"

# Test router pass-through
state = {"top_similarity": 0.85, "retrieved_chunks": [...]}
result = router_node(state)
assert result["intent"] == "answer"
```

### Integration Tests
```python
# Full pipeline test with a sample PDF
store = ingest_pdfs(["tests/fixtures/sample_manual.pdf"])
graph = build_graph(store)
state = graph.invoke({"query": "What is the return policy?", "node_trace": []})
assert state["final_answer"]
assert "retriever" in state["node_trace"]
```

### Sample Queries for Evaluation

| Query | Expected Route | Pass Criteria |
|---|---|---|
| "What is the return policy?" | generator | Answer references policy section |
| "How do I reset my password?" | generator | Answer contains step-by-step instructions |
| "What is the meaning of life?" | hitl | Escalation triggered (off-topic) |
| "asdfjkl qwerty" | hitl | Escalation triggered (gibberish) |
| "Tell me about [topic not in PDF]" | hitl | Escalation triggered (missing context) |

### Threshold Calibration
1. Collect 50 representative queries.
2. Label each as "should answer" or "should escalate".
3. Run at thresholds 0.60, 0.65, 0.70, 0.75, 0.80.
4. Choose the threshold that maximises F1 on the "should answer" class.

---

## 9. Future Enhancements

### Multi-Document Support
Already supported — `ingest_pdfs()` accepts a list of paths. Add a metadata filter to `VectorStore.query()` to restrict retrieval to a specific document.

### Feedback Loop
Log every query, retrieved chunks, and final answer. When a human agent provides a HITL response, store it as a high-quality Q&A pair and re-ingest it as a synthetic document to improve future retrieval.

### Memory Integration
Add a `ConversationBufferMemory` to `GraphState` to support multi-turn conversations. The retriever node would prepend recent turns to the query for context-aware retrieval.

### Deployment
```
FastAPI wrapper → Docker container → Kubernetes (HPA on CPU/memory)
ChromaDB → Chroma Cloud or Pinecone (managed vector store)
LLM → OpenAI API or self-hosted vLLM
HITL → Zendesk webhook or Slack bot
```

### Evaluation Framework
Integrate RAGAS (Retrieval-Augmented Generation Assessment) to automatically score:
- **Faithfulness**: Is the answer grounded in the retrieved context?
- **Answer Relevancy**: Does the answer address the question?
- **Context Recall**: Were the right chunks retrieved?
