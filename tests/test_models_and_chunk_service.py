from dataclasses import FrozenInstanceError

import pytest

from model import Chunk, ConversationHistory
from pipeline.components.chunk_service import ChunkService


def test_chunk_fields_repr_and_immutability():
    chunk = Chunk(1, "hello", "source.txt")
    assert (chunk.id, chunk.content, chunk.source_file) == (1, "hello", "source.txt")
    assert repr(chunk) == "Chunk{id=1, content='hello'}"
    with pytest.raises(FrozenInstanceError):
        chunk.content = "changed"


def test_conversation_history_model_fields():
    item = ConversationHistory(conversation_id="c1", conversation_title="Title", history="[]")
    assert item.conversation_id == "c1"
    assert item.conversation_title == "Title"
    assert item.history == "[]"


def test_chunk_service_split_sets_source_and_sequential_ids():
    service = ChunkService(vector_store_service=None)
    text = "First paragraph. " * 30
    chunks = service.split(text, "doc.txt")
    assert len(chunks) > 1
    assert [chunk.id for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.source_file == "doc.txt" for chunk in chunks)
    assert "First paragraph." in "".join(c.content for c in chunks)


def test_chunk_service_defaults_missing_source():
    chunks = ChunkService(None).split("some text", None)
    assert chunks and chunks[0].source_file == "unknown"


def test_chunk_service_retrieve_delegates():
    class Store:
        def retrieve(self, query, top_k):
            return [query, top_k]
    assert ChunkService(Store()).retrieve("hello") == ["hello", 3]
