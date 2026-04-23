# RAG Customer Support Assistant

A production-grade Retrieval-Augmented Generation (RAG) system with LangGraph workflow orchestration and Human-in-the-Loop (HITL) escalation.

---

## Project Structure

```
rag_assistant/
├── config.py                  # Central configuration (all tuneable params)
├── main.py                    # CLI entry point
├── ingestion/
│   ├── loader.py              # PDF → RawDocument (PDFPlumber)
│   ├── chunker.py             # RawDocument → Chunk (sliding window)
│   ├── embedder.py            # Text → vectors (OpenAI or HuggingFace)
│   ├── vector_store.py        # ChromaDB wrapper (upsert + cosine search)
│   └── pipeline.py            # Orchestrates full ingestion flow
├── graph/
│   ├── state.py               # GraphState TypedDict schema
│   ├── nodes.py               # Node A (Retriever), B (Router), C (Generator), D (HITL)
│   └── workflow.py            # LangGraph StateGraph assembly
└── hitl/
    └── handler.py             # CLI HITL handler (blocks for human input)

docs/
├── HLD.md                     # High-Level Design + Mermaid architecture diagram
├── LLD.md                     # Low-Level Design + class structures + routing logic
└── TechDoc.md                 # Technical documentation + trade-offs + testing
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For offline use (no API keys needed), the defaults use HuggingFace models.  
For OpenAI, set environment variables:

```bash
export EMBEDDING_PROVIDER=openai
export OPENAI_API_KEY=sk-...
```

### 2. Ingest a PDF knowledge base

```bash
python -m rag_assistant.main ingest --pdf path/to/your_manual.pdf
```

Multiple PDFs:
```bash
python -m rag_assistant.main ingest --pdf manual.pdf faq.pdf policy.pdf
```

### 3. Start the interactive chat

```bash
python -m rag_assistant.main chat
```

### 4. Single query (non-interactive)

```bash
python -m rag_assistant.main query --text "How do I reset my password?"
```

### 5. Reset the knowledge base

```bash
python -m rag_assistant.main reset
```

---

## Configuration

All settings are in `rag_assistant/config.py` and can be overridden via environment variables:

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_PROVIDER` | `huggingface` | `openai` or `huggingface` |
| `OPENAI_API_KEY` | *(empty)* | Required if using OpenAI |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB storage directory |
| `LOG_LEVEL` | `INFO` | Python logging level |

Tuneable parameters (edit `config.py`):

| Parameter | Default | Effect |
|---|---|---|
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `TOP_K_RESULTS` | `5` | Chunks retrieved per query |
| `SIMILARITY_THRESHOLD` | `0.70` | Below this → HITL escalation |
| `HITL_TIMEOUT_SECONDS` | `300` | Seconds to wait for human input |

---

## LangGraph Workflow

```
START → retriever → router ──[answer]──► generator → END
                         └──[escalate]─► hitl      → END
```

**Escalation triggers:**
- No chunks retrieved (query outside knowledge base)
- `top_similarity < 0.70` (low retrieval confidence)
- LLM expresses uncertainty in its response
- Any upstream error

When escalation occurs, the CLI pauses and prompts a human agent for a response.

---

## Deliverables

| File | Description |
|---|---|
| `docs/HLD.md` | System overview, Mermaid architecture diagram, scalability plan |
| `docs/LLD.md` | Class structures, state schema, routing logic, error handling |
| `docs/TechDoc.md` | RAG explanation, design decisions, trade-offs, testing strategy |

---

## Stack

- **Python 3.10+**
- **LangGraph** — stateful graph workflow with conditional routing
- **LangChain** — LLM abstraction layer
- **ChromaDB** — persistent vector store with cosine similarity
- **PDFPlumber** — accurate PDF text extraction
- **sentence-transformers** — offline HuggingFace embeddings
- **OpenAI** — optional cloud embeddings and LLM
