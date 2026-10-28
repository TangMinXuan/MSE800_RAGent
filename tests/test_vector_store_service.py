import pytest

from model import Chunk
from pipeline.components.vector_store_service import CHUNK_ID, SOURCE_FILE, VectorStoreService


class FakeChroma:
    def __init__(self, matches=None):
        self.added = None
        self.deleted = []
        self.matches = matches or []
        self.search_args = None

    def add_segments(self, segments):
        self.added = segments

    def search(self, query, max_results, min_score):
        self.search_args = (query, max_results, min_score)
        return self.matches

    def delete_by_source(self, source_file):
        self.deleted.append(source_file)


def test_upsert_converts_chunks_to_segments():
    db = FakeChroma()
    VectorStoreService(db).upsert([Chunk(4, "hello", "a.txt")])
    assert db.added == [{"text": "hello", "metadata": {CHUNK_ID: 4, SOURCE_FILE: "a.txt"}}]


def test_retrieve_maps_matches_to_chunks_and_uses_threshold():
    db = FakeChroma([{"metadata": {CHUNK_ID: "2", SOURCE_FILE: "a.txt"}, "text": "found", "score": 0.9}])
    result = VectorStoreService(db).retrieve("question", 5)
    assert result == [Chunk(2, "found", "a.txt")]
    assert db.search_args == ("question", 5, 0.3)


def test_delete_by_source_rejects_blank_source():
    with pytest.raises(ValueError, match="must not be blank"):
        VectorStoreService(FakeChroma()).delete_by_source(" ")


def test_delete_by_source_delegates():
    db = FakeChroma()
    VectorStoreService(db).delete_by_source("a.txt")
    assert db.deleted == ["a.txt"]
