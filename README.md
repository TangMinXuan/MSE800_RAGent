# RAGent

A multi-knowledge-base question-answering system built on Retrieval-Augmented Generation (RAG) and a ReAct agent.
One architecture serves very different organisations. The demo ships with two knowledge bases:

- **Māori Honey**: product information for a traditional business.
- **TangFan DB**: product and onboarding documentation for an IT team.

## Features

- **Intent recognition**: a ReAct agent (LangGraph) answers general questions directly and routes document questions to the right knowledge base via tools.
- **Context-aware query rewriting**: follow-ups such as "What about its price?" are rewritten into standalone queries.
- **Persistent conversation memory**: stored in PostgreSQL, 20-message sliding window, auto-generated titles.
- **Streaming output**: token-by-token answers over Server-Sent Events (SSE).
- **Extensible design**: a new knowledge base needs only a new Chroma collection and one tool registration.
- **Idempotent ingestion**: deterministic chunk IDs and delete-before-upsert, so re-ingesting never creates duplicates.

## Architecture

| Layer | Technology |
|---|---|
| Frontend | React (Node.js, port 3000) |
| Backend | Python, FastAPI (port 7070), LangChain, LangGraph |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Vector DB | ChromaDB (HTTP, default `http://localhost:8000`) |
| Relational DB | PostgreSQL (conversation history) |
| LLM | `openai/gpt-4o-mini` via OpenRouter |

Diagrams (Mermaid) are in the repository root:

- `architecture.mmd`: system architecture
- `ingestion_flow.mmd`: ingestion pipeline (load → split → embed → store)
- `rag_flow.mmd`: intent recognition + RAG pipeline (history → rewrite → retrieve → build context → LLM)

## Project Structure

```
api/            FastAPI server (SSE endpoints)
pipeline/       LangGraph ingestion and RAG pipelines, plus components
infra/          Chroma, PostgreSQL and LLM clients
model/          Data models
util/           Helpers (env, file, text, retry, ...)
resources/      Sample knowledge documents
tang_fan_agent.py   ReAct agent, system prompt, streaming and memory
agent_tool.py       Knowledge-base tools exposed to the agent
```

## Prerequisites

- Python 3.10+
- Node.js (for the frontend, run separately)
- A running ChromaDB server
- A running PostgreSQL database with the `conversation_history` table created
- An OpenRouter API key

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment variables

| Variable | Description | Default |
|---|---|---|
| `API_KEY` | OpenRouter API key | required |
| `POSTGRES_URL` | SQLAlchemy URL, e.g. `postgresql+psycopg://localhost:5432/ragent` | required |
| `POSTGRES_USER` | PostgreSQL user | required |
| `POSTGRES_PASSWORD` | PostgreSQL password | required |
| `CHROMA_URL` | ChromaDB server URL | `http://localhost:8000` |

## Run

Start each service from the command line, in this order.

1. **ChromaDB**

   ```bash
   chroma run --path ./chroma_data --port 8000
   ```

2. **PostgreSQL**: start your local instance and make sure the database and table exist.

3. **Ingest the knowledge documents** (once, and again whenever a document changes). Paths in `main()` of `pipeline/ingestion_pipeline.py` are currently absolute, so update them to your machine first.

   ```bash
   python -m pipeline.ingestion_pipeline
   ```

4. **Backend** (port 7070)

   ```bash
   python -m api.api_server
   ```

5. **Frontend** (port 3000): start the React app from the frontend project with its usual `npm install` and `npm run dev` / `npm start`.

Then open <http://localhost:3000>.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/api/chat/stream` | Streaming chat over SSE |
| POST | `/api/chat` | Non-streaming chat (currently a stub) |

SSE events: `conversationId`, `delta`, `conversationTitle`, `done`, `error`.

## Demo questions

- General: "What is RAG?" (answered by the LLM directly)
- TangFan DB: "What algorithm did Master Tang invent?"
- Māori Honey: "How should I store Manuka honey?", then the follow-up "What about its price?"

## Roadmap

- Semantic and heading-aware chunking, parent-child chunking
- Reranking and hybrid search (BM25 + vector)
- Reuse the agent and database connections across requests
- Docker Compose for one-command startup

## Team

A four-member multicultural team (China, India, Bangladesh) for the MSE800 course.
