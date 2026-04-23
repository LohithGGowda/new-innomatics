# Low-Level Design (LLD)
## RAG-Based Customer Support Assistant

---

## 1. Module-Level Design

### 1.1 Document Processing Module — `ingestion/loader.py`

```
load_pdf(pdf_path) -> List[RawDocument]
load_pdfs(pdf_paths) -> List[RawDocument]
```

- Opens each PDF with `pdfplumber.open()`.
- Iterates pages, calls `page.extract_text()`.
- Skips pages with no extractable text (logs a warning).
- Raises `FileNotFoundError` / `ValueError` on hard failures.

### 1.2 Chunking Module — `ingestion/chunker.py`

```
_split_text(text, chunk_size, overlap) -> List[str]
chunk_documents(documents, chunk_size, overlap) -> List[Chunk]
```

- Sliding window: `step = chunk_size - overlap`.
- Each chunk carries `chunk_id = "<stem>_p<page>_c<idx>"` for deduplication.
- Configurable via `config.chunk_size` (default 512) and `config.chunk_overlap` (default 64).

### 1.3 Embedding Module — `ingestion/embedder.py`

```
EmbeddingModel (ABC)
  ├── OpenAIEmbeddingModel
  └── HuggingFaceEmbeddingModel

get_embedding_model() -> EmbeddingModel
```

- Factory function reads `config.embedding_provider` and returns the correct backend.
- Both backends expose identical `embed_texts(List[str])` and `embed_query(str)` methods.

### 1.4 Vector Storage Module — `ingestion/vector_store.py`

```
VectorStore
  ├── add_chunks(chunks, batch_size=64)
  ├── query(query_text, top_k) -> List[RetrievedChunk]
  ├── count() -> int
  └── reset()
```

- Uses `chromadb.PersistentClient` with `hnsw:space = cosine`.
- `add_chunks` is idempotent — existing `chunk_id`s are skipped.
- Distance-to-similarity conversion: `similarity = 1 - cosine_distance`.

### 1.5 Retrieval Module — `graph/nodes.py::retriever_node`

- Calls `VectorStore.query()`.
- Builds a numbered context block from returned chunks.
- Writes `retrieved_chunks`, `top_similarity`, `synthesized_context` to state.

### 1.6 Query Processing Module — `graph/nodes.py::generator_node`

- Constructs a grounded prompt with context and query.
- Dispatches to OpenAI or HuggingFace LLM based on config.
- Detects uncertainty phrases in LLM output and re-routes to HITL.

### 1.7 Graph Execution Module — `graph/workflow.py`

- Builds `StateGraph(GraphState)`.
- Registers 4 nodes, sets entry point, wires edges.
- Compiles to a callable via `graph.compile()`.

### 1.8 HITL Module — `hitl/handler.py`

- `handle_escalation(query, reason, llm_draft, timeout)` blocks for human input.
- Uses `signal.SIGALRM` for timeout on POSIX systems.
- Returns human response string; caller writes it to `GraphState`.

---

## 2. Data Structures

### 2.1 RawDocument
```python
@dataclass
class RawDocument:
    source: str          # absolute PDF path
    page_number: int     # 1-based
    text: str            # raw page text
    metadata: dict       # {"source", "page", "total_pages"}
```

### 2.2 Chunk
```python
@dataclass
class Chunk:
    chunk_id: str        # "<stem>_p<page>_c<idx>"
    text: str
    source: str
    page_number: int
    chunk_index: int
    metadata: dict       # includes chunk_id, chunk_index, source, page
```

### 2.3 RetrievedChunk
```python
@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    source: str
    page_number: int
    similarity: float    # cosine similarity ∈ [0, 1]
    metadata: dict
```

### 2.4 GraphState (TypedDict)
```python
class GraphState(TypedDict, total=False):
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
```

---

## 3. Workflow Design (LangGraph)

### Node Responsibilities

| Node | Input Keys | Output Keys | Side Effects |
|---|---|---|---|
| `retriever` | `query` | `retrieved_chunks`, `top_similarity`, `synthesized_context` | ChromaDB query |
| `router` | `top_similarity`, `retrieved_chunks`, `error` | `intent`, `escalation_reason` | None |
| `generator` | `query`, `synthesized_context` | `llm_response`, `final_answer` | LLM API call |
| `hitl` | `query`, `escalation_reason` | *(state marker)* | Blocks for human input |

### Edge Map

```
START
  └─► retriever
        └─► router
              ├─► [intent="answer"]   generator ──► END
              └─► [intent="escalate"] hitl      ──► END
```

### State Transitions

```
Initial: { query, node_trace: [] }

After retriever:
  + retrieved_chunks, top_similarity, synthesized_context

After router:
  + intent, escalation_reason

After generator (happy path):
  + llm_response, final_answer

After hitl + handler:
  + human_response, final_answer
```

---

## 4. Conditional Routing Logic

```python
def _route_after_router(state: GraphState) -> Literal["generator", "hitl"]:
    if state.get("intent") == "answer":
        return "generator"
    return "hitl"
```

### Escalation Criteria (evaluated in `router_node`)

| Condition | Trigger |
|---|---|
| `state["error"]` is set | Upstream failure (retrieval error) |
| `len(retrieved_chunks) == 0` | No documents in knowledge base match the query |
| `top_similarity < 0.70` | Low retrieval confidence |

### Generator Re-escalation (evaluated in `generator_node`)

If the LLM response contains any of:
- `"i'm not sure"`, `"i don't know"`, `"cannot answer"`,
  `"not enough information"`, `"contact a human"`

→ `intent` is overridden to `"escalate"` and the HITL handler is invoked by the runner.

---

## 5. HITL Design

### When Escalation is Triggered
1. Router detects low similarity or missing context.
2. Generator detects LLM uncertainty.
3. Any upstream node sets `state["error"]`.

### Escalation Flow
```
router_node sets intent="escalate"
  └─► hitl_node marks state (node_trace += "hitl")
        └─► graph returns to runner
              └─► runner calls handle_escalation()
                    └─► CLI prints escalation banner
                          └─► Human agent types response
                                └─► runner writes human_response → final_answer
```

### Timeout Behaviour
- Default: 300 seconds (configurable via `config.hitl_timeout_seconds`).
- On timeout: returns a canned "unable to reach agent" message.
- On `KeyboardInterrupt`: returns a "session interrupted" message.

### Human Response Integration
The runner (`main.py::run_query`) checks `"hitl" in node_trace` after graph execution and calls `handle_escalation()`. The returned string is written directly into `final_answer`.

---

## 6. API / Interface Design

### CLI Input Format
```
python -m rag_assistant.main ingest --pdf <path> [<path> ...]
python -m rag_assistant.main chat
python -m rag_assistant.main query --text "<question>"
python -m rag_assistant.main reset
```

### Programmatic Input
```python
initial_state: GraphState = {
    "query": "How do I reset my password?",
    "node_trace": [],
}
final_state = graph.invoke(initial_state)
answer = final_state["final_answer"]
```

### Output Format
```python
{
    "query": str,
    "final_answer": str,
    "intent": "answer" | "escalate",
    "top_similarity": float,
    "node_trace": List[str],   # e.g. ["retriever", "router", "generator"]
    "escalation_reason": Optional[str],
}
```

---

## 7. Error Handling

| Scenario | Handling |
|---|---|
| PDF not found | `FileNotFoundError` raised in loader; logged and skipped in `load_pdfs`. |
| PDF has no text | `ValueError` raised; logged. |
| ChromaDB unavailable | Exception propagates to `retriever_node`; sets `state["error"]`; router escalates. |
| No chunks retrieved | Router detects empty list; escalates with "No relevant documents found." |
| LLM API failure | Exception caught in `generator_node`; sets `state["error"]`; escalates. |
| LLM uncertainty | Detected via phrase matching; re-routes to HITL. |
| HITL timeout | `TimeoutError` caught; returns canned message. |
| Empty human response | Replaced with default "agent notified" message. |
