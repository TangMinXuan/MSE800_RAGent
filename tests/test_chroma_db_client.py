import pytest

from infra.database.chroma_db_client import ChromaDbClient, DEFAULT_COLLECTION, DEFAULT_URL


class FakeEmbedding:
    def __init__(self):
        self.document_calls = []
        self.query_calls = []

    def embed_documents(self, texts):
        self.document_calls.append(texts)
        return [[0.1, 0.2] for _ in texts]

    def embed_query(self, query):
        self.query_calls.append(query)
        return [0.1, 0.2]


class FakeCollection:
    name = "test_collection"
    metadata = {"hnsw:space": "cosine"}
    configuration = {}

    def __init__(self):
        self.upsert_call = None
        self.query_call = None
        self.delete_call = None

    def upsert(self, **kwargs):
        self.upsert_call = kwargs

    def query(self, **kwargs):
        self.query_call = kwargs
        return {"documents": [["match", "low score"]], "metadatas": [[{"id": 1}, {}]], "distances": [[0.1, 0.9]]}

    def delete(self, **kwargs):
        self.delete_call = kwargs


class FakeChromaClient:
    def __init__(self, collection):
        self.collection = collection
        self.requested = None

    def get_or_create_collection(self, **kwargs):
        self.requested = kwargs
        return self.collection


def make_client(collection=None, embedding=None):
    collection = collection or FakeCollection()
    client = FakeChromaClient(collection)
    embedding = embedding or FakeEmbedding()
    return ChromaDbClient("http://localhost:8000", "test_collection", embedding, client), client, collection, embedding


def test_from_environment_uses_defaults(monkeypatch):
    monkeypatch.delenv("CHROMA_URL", raising=False)
    monkeypatch.delenv("CHROMA_COLLECTION", raising=False)
    assert DEFAULT_URL == "http://localhost:8000"
    assert DEFAULT_COLLECTION == "default_collection"
    # Constructor is tested separately with a fake Chroma client to avoid network access.


def test_constructor_requests_cosine_collection():
    _, client, _, _ = make_client()
    assert client.requested == {"name": "test_collection", "metadata": {"hnsw:space": "cosine"}}


def test_constructor_rejects_non_cosine_collection():
    collection = FakeCollection()
    collection.metadata = {"hnsw:space": "l2"}
    with pytest.raises(RuntimeError, match="expected 'cosine'"):
        make_client(collection=collection)


def test_add_segments_embeds_and_upserts_with_stable_ids():
    db, _, collection, embedding = make_client()
    segments = [{"text": "hello", "metadata": {"source_file": "a.txt"}}]
    db.add_segments(segments)
    assert embedding.document_calls == [["hello"]]
    assert collection.upsert_call["documents"] == ["hello"]
    assert collection.upsert_call["metadatas"] == [{"source_file": "a.txt"}]
    assert collection.upsert_call["ids"] == [db._make_id(0, "hello", {"source_file": "a.txt"})]
    assert db._make_id(0, "hello", {"a": 1, "b": 2}) == db._make_id(0, "hello", {"b": 2, "a": 1})


def test_add_empty_segments_is_noop():
    db, _, collection, embedding = make_client()
    db.add_segments([])
    assert collection.upsert_call is None
    assert embedding.document_calls == []


def test_search_converts_cosine_distance_and_filters_low_scores():
    db, _, collection, embedding = make_client()
    result = db.search("question", 4, 0.5)
    assert result == [{"text": "match", "metadata": {"id": 1}, "score": pytest.approx(0.9)}]
    assert embedding.query_calls == ["question"]
    assert collection.query_call["n_results"] == 4


def test_delete_by_source_uses_metadata_filter():
    db, _, collection, _ = make_client()
    db.delete_by_source("source.txt")
    assert collection.delete_call == {"where": {"source_file": "source.txt"}}
