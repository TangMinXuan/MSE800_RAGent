from pathlib import Path

import pytest

from model import Chunk
from pipeline.ingestion_pipeline import IngestionPipeline


class FakeChunkService:
    def __init__(self):
        self.calls = []

    def split(self, text, source_file):
        self.calls.append((text, source_file))
        return [Chunk(0, text[:20], source_file)] if text else []


class FakeVectorStore:
    def __init__(self):
        self.deleted = []
        self.upserts = []

    def delete_by_source(self, source):
        self.deleted.append(source)

    def upsert(self, chunks):
        self.upserts.append(chunks)


def make_pipeline():
    chunk, store = FakeChunkService(), FakeVectorStore()
    return IngestionPipeline(chunk, store), chunk, store


def test_ingest_text_file_splits_replaces_and_stores_chunks(tmp_path):
    path = tmp_path / "knowledge.txt"
    path.write_text("Useful knowledge text", encoding="utf-8")
    pipeline, chunk, store = make_pipeline()
    count = pipeline.ingest(path)
    assert count == 1
    assert chunk.calls[0][0] == "Useful knowledge text"
    source = str(path.resolve())
    assert chunk.calls[0][1] == source
    assert store.deleted == [source]
    assert len(store.upserts[0]) == 1


def test_ingest_missing_file_raises_value_error(tmp_path):
    pipeline, _, _ = make_pipeline()
    with pytest.raises(ValueError, match="Document file does not exist"):
        pipeline.ingest(tmp_path / "missing.txt")
    with pytest.raises(ValueError, match="Document file does not exist"):
        pipeline.ingest(None)


def test_load_document_rejects_missing_file_in_graph_state(tmp_path):
    pipeline, _, _ = make_pipeline()
    with pytest.raises(ValueError, match="A document file is required"):
        pipeline._load_document({})
    with pytest.raises(FileNotFoundError, match="Document file not found"):
        pipeline._load_document({"file": tmp_path / "missing.txt"})


def test_load_document_reads_utf8_text(tmp_path):
    path = tmp_path / "maori.txt"
    path.write_text("Kia ora Māori", encoding="utf-8")
    pipeline, _, _ = make_pipeline()
    state = pipeline._load_document({"file": path})
    assert state["document_text"] == "Kia ora Māori"
    assert state["source_file"] == str(path.resolve())


def test_pipeline_rejects_invalid_graph_states():
    pipeline, _, _ = make_pipeline()
    with pytest.raises(RuntimeError, match="document was not loaded"):
        pipeline._split_document({})
    with pytest.raises(RuntimeError, match="source file was not retained"):
        pipeline._split_document({"document_text": "text"})
    with pytest.raises(RuntimeError, match="document was not split"):
        pipeline._store_chunks({})
