# AI Research Paper Intelligence & Multimodal Analysis System

A production-oriented, modular research assistant that lets you upload one or
more research papers (PDF) and ask questions about them — single-paper
Q&A, cross-paper comparison, figure/table understanding, and tool-assisted
calculations — backed by RAG, a multi-agent LangGraph workflow, and MCP-based
tool orchestration.

## Architecture

```
USER
  ↓
STREAMLIT UI  ──(HTTP)──►  FASTAPI  ──►  QUERY ROUTER (/query)
                                            ↓
                                       PLANNER AGENT
                                            ↓
                                  LANGGRAPH ORCHESTRATOR
                                            ↓
                    ┌───────────────────────────────────────┐
                    │  Retriever Agent                       │
                    │  Research Analyst Agent                │
                    │  Multimodal Analysis Agent              │
                    │  Tool Agent → MCP Client → MCP Server   │
                    │              → Tool → Result            │
                    └───────────────────────────────────────┘
                                            ↓
                                     REVIEWER AGENT
                                    ↙ approved   ↘ rejected (bounded retries)
                                 END           back to Retriever
```

Every agent is a plain Python module (`app/agents/*.py`). LangGraph
(`app/graph/*.py`) is the *only* place that wires them into a stateful,
conditionally-routed graph — agents never call each other directly.

## Project layout

```
research-intelligence/
├── app/
│   ├── main.py            # FastAPI app (upload, list/clear docs, /query)
│   ├── config.py          # Pydantic settings loaded from .env
│   ├── agents/             # Planner, Retriever, Researcher, Multimodal, Tool, Reviewer
│   ├── graph/               # LangGraph state / nodes / workflow
│   ├── rag/                  # parsing, chunking, embeddings, vector store, retrieval
│   ├── multimodal/            # figure/table extraction + figure description
│   ├── mcp/                    # MCP servers + client (separate from agents)
│   ├── tools/                   # calculator, document tools, web search
│   └── models/schemas.py         # all Pydantic models
├── ui/app.py                       # Streamlit chat UI
├── data/{papers,images,processed}   # local storage (git-ignored)
├── vectorstore/                      # Chroma persistence (git-ignored)
├── tests/                             # pytest unit tests
├── main.py                             # `python main.py api|ui|ingest`
├── requirements.txt
└── .env.example
```

## How it works

### 1. Ingestion pipeline (`app/rag/ingestion.py`)

```
PDF → validate → parse (PyMuPDF, section detection)
    → chunk (LangChain recursive splitter, page/section-aware)
    → extract figures (PyMuPDF) → describe (vision model)
    → extract tables (pdfplumber → markdown)
    → embed everything → write to vector store (Chroma)
```

Every stored item — text chunk, figure, table — keeps full provenance
(`document_id`, `filename`, `page_number`, `section`, `content_type`, ...),
which is what makes accurate citations possible later.

### 2. RAG + multi-agent query flow (`app/graph/workflow.py`)

1. **Planner** classifies the query (simple/complex), decides whether
   multimodal analysis or a tool is needed, and produces an `ExecutionPlan`.
2. **Retriever** embeds the query and pulls top-k text (and, when needed,
   figure/table) context from the vector store.
3. **Research Analyst** produces a grounded draft answer from the text
   context — summarizing, comparing, extracting methodology/datasets/results.
4. **Multimodal Agent** (only when the plan requires it) combines text +
   figure + table context to answer questions about visual content.
5. **Tool Agent** (only when the plan requires it) calls a tool — e.g. the
   calculator — via the **MCP client → MCP server → tool** path.
6. **Reviewer** checks the draft against retrieved evidence. If unsupported
   claims or missing evidence are detected, it rejects and the graph loops
   back to the Retriever — bounded by `MAX_REVIEW_RETRIES` in `.env` to
   avoid infinite loops.
7. On approval, citations are built from the actual retrieved metadata
   (never invented) and returned to the UI.

### 3. MCP tool orchestration

`app/mcp/servers/*.py` each expose one focused tool set (`calculate`,
`search_documents` / `get_document_page`, `search_web`) as a standalone MCP
server communicating over stdio. `app/mcp/client.py` is the only code that
spawns these servers and calls tools on them — the Tool Agent never touches
`app/tools/*.py` directly, keeping the MCP layer cleanly separated from the
LangGraph agents as required by the architecture.

## Setup

### 1. Prerequisites

- Python 3.11+
- An OpenAI API key
- (Optional) PostgreSQL if you want to persist app metadata there instead of
  relying on the vector store alone
- (Optional) a SerpAPI key if you want the `search_web` tool to actually hit
  the web instead of returning a "not configured" message

### 2. Install

```bash
cd research-intelligence
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# then edit .env and set OPENAI_API_KEY at minimum
```

Key variables (see `.env.example` for the full list):

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | required — used for embeddings, chat, and vision |
| `OPENAI_CHAT_MODEL` | model used by all text agents (default `gpt-4o-mini`) |
| `OPENAI_VISION_MODEL` | model used to describe extracted figures |
| `OPENAI_EMBEDDING_MODEL` | embedding model (default `text-embedding-3-small`) |
| `VECTOR_STORE_PROVIDER` | `chroma` today; `qdrant` reserved for later |
| `CHROMA_PERSIST_DIR` | local path for the Chroma database |
| `MAX_REVIEW_RETRIES` | bounds the Reviewer → Retriever loop |
| `TOP_K_RETRIEVAL` | number of chunks retrieved per query |
| `SERPAPI_API_KEY` | optional, enables the `search_web` tool |

### 4. Run

In two terminals:

```bash
# Terminal 1 — backend
python main.py api
# equivalent to: uvicorn app.main:app --reload

# Terminal 2 — UI
python main.py ui
# equivalent to: streamlit run ui/app.py
```

Open the Streamlit URL it prints (typically `http://localhost:8501`), upload
one or more PDFs from the sidebar, click **Process papers**, then ask
questions in the chat box.

### 5. Ingest from the CLI (optional)

```bash
python main.py ingest path/to/paper.pdf
```

### 6. Run the MCP servers standalone (optional, for inspection/debugging)

```bash
python -m app.mcp.servers.calculator_server
python -m app.mcp.servers.document_server
python -m app.mcp.servers.search_server
```

Each speaks MCP over stdio and can be attached to any MCP-compatible client
(e.g. Claude Desktop, the `mcp` CLI inspector) independently of this app.

### 7. Run tests

```bash
pytest -q
```

Tests that don't require network access (calculator, chunking, schemas)
run offline. Tests exercising embeddings/LLM calls require `OPENAI_API_KEY`
to be set and will make real API calls.

## Example questions

- "Summarize this paper."
- "What methodology is used and what dataset was it evaluated on?"
- "Compare the methodologies and datasets used in all uploaded papers."
- "Explain Figure 3." / "What does this architecture diagram show?"
- "Calculate the average BLEU score across the papers."

## Design notes / extension points

- **Swapping to Qdrant**: implement `QdrantVectorStore(VectorStore)` in
  `app/rag/vector_store.py` and select it via `VECTOR_STORE_PROVIDER=qdrant`
  in `.env` — no other file needs to change, since every caller depends only
  on the `VectorStore` interface.
- **Adding a new agent capability**: add the function to `app/agents/`, wire
  it into `app/graph/nodes.py` as a new node, and add it to
  `app/graph/workflow.py`'s edges/conditional routing.
- **Adding a new MCP tool**: implement it in `app/tools/`, expose it from a
  new or existing server in `app/mcp/servers/`, and add a matching wrapper
  function in `app/mcp/client.py` for the Tool Agent to call.
- **PostgreSQL**: `app/config.py` exposes a ready-to-use `postgres_dsn`
  property; wire up a SQLAlchemy session where application-level metadata
  (e.g. user accounts, upload history) is needed. It is optional today
  because document metadata already lives alongside vectors in Chroma.

## Limitations (by design, for a portfolio-scale project)

- The MCP client spawns a fresh subprocess per tool call rather than pooling
  long-lived sessions — simple and robust, but not optimized for high
  throughput.
- Section detection and table extraction use heuristics (regex header
  matching, `pdfplumber`'s ruled-table detection) rather than a full
  document-layout model, so results vary with PDF formatting quality.
- Figure captions are matched heuristically to nearby "Figure N" text on the
  same page rather than via true spatial layout analysis.
