import pytest

from model import Chunk
from pipeline.rag_pipeline import RagPipeline, RagResult


class FakeChunkService:
    def __init__(self, chunks=None):
        self.chunks = chunks if chunks is not None else [Chunk(1, "reference", "doc.txt")]
        self.queries = []

    def retrieve(self, query):
        self.queries.append(query)
        return self.chunks


class FakePromptService:
    def __init__(self):
        self.rewrite_calls = []
        self.context_calls = []

    def rewrite_query(self, history, question):
        self.rewrite_calls.append((history, question))
        return "rewritten: " + question

    def build_retrieved_context(self, chunks):
        self.context_calls.append(chunks)
        return "CONTEXT: " + ",".join(chunk.content for chunk in chunks)


def test_invoke_runs_rewrite_retrieve_and_context_steps():
    chunks = [Chunk(1, "reference", "doc.txt")]
    chunk_service, prompt_service = FakeChunkService(chunks), FakePromptService()
    pipeline = RagPipeline(chunk_service, prompt_service)
    result = pipeline.invoke("What is this?", [])
    assert isinstance(result, RagResult)
    assert result.context == "CONTEXT: reference"
    assert result.chunks == chunks
    assert prompt_service.rewrite_calls == [([], "What is this?")]
    assert chunk_service.queries == ["rewritten: What is this?"]
    assert prompt_service.context_calls == [chunks]


@pytest.mark.parametrize("question", [None, "", "   "])
def test_invoke_rejects_missing_or_blank_question(question):
    pipeline = RagPipeline(FakeChunkService(), FakePromptService())
    with pytest.raises(ValueError, match="Question must not be blank"):
        pipeline.invoke(question, None)


def test_invoke_accepts_none_history_as_empty_history():
    prompt_service = FakePromptService()
    result = RagPipeline(FakeChunkService(), prompt_service).invoke("question", None)
    assert prompt_service.rewrite_calls[0][0] == []
    assert result.context.startswith("CONTEXT:")
