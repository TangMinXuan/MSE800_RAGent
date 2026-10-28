# Unit Testing Guide — MSE800 RAGent

## What is included

The `tests/` directory contains pytest tests for the current Python implementation:

- utility functions (collections, environment variables, files, JSON, retry, text, time)
- `Chunk` and `ConversationHistory` models and `ChunkService`
- `PromptService`
- `VectorStoreService`
- `RagPipeline` graph flow with fake services
- `IngestionPipeline` with fake chunk/vector-store services
- FastAPI health/SSE behaviour and current chat endpoint behaviour
- `ConversationMemory` / `ChatMemory` logic
- PostgreSQL helper validation
- Chroma client behaviour with fake collection/client objects
- `AgentTool`
- `TangFanAgent` message handling and streaming with a fake graph

External services (OpenAI/OpenRouter, PostgreSQL, and Chroma server) are not called by these tests. Tests use fakes/mocks for service boundaries where practical.

## Install and run on Windows

Open PowerShell in the project root, then run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

If PowerShell blocks activation, you can run the virtual environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

To run one file:

```powershell
python -m pytest tests/test_rag_pipeline.py -v
```

To create a coverage report (optional):

```powershell
python -m pip install pytest-cov
python -m pytest --cov=. --cov-report=term-missing
```

## Important code observation

`POST /api/chat` currently returns hard-coded placeholder values (`conversationId="blanket for test"`, `llmResp="1+1=2"`) rather than calling the agent. A test characterizes this current behaviour so it is visible; it does **not** mean this endpoint is functionally complete. Before production or acceptance testing, implement the intended non-streaming chat flow and then update the test to assert the real expected behaviour.

## Notes

- These are unit tests, not full end-to-end/acceptance tests.
- Tests should be run in the same Python version/environment used by the team.
- If a test fails after installing dependencies, read the first traceback carefully; package-version mismatches may need to be aligned with the team's working environment.
