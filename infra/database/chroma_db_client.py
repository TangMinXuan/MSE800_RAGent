import os
import uuid
from typing import Any
from urllib.parse import urlparse

import chromadb
from langchain_core.embeddings import Embeddings

DEFAULT_URL = "http://localhost:8000"
DEFAULT_COLLECTION = "default_collection"


class ChromaDbClient:
    def __init__(self, base_url: str, collection_name: str, embedding_model: Embeddings, client: Any = None):
        self.embedding_model = embedding_model
        if client is None:
            parsed = urlparse(base_url)
            client = chromadb.HttpClient(
                host=parsed.hostname or "localhost",
                port=parsed.port or (443 if parsed.scheme == "https" else 8000),
                ssl=parsed.scheme == "https",
            )
        self._client = client
        self.collection = client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )
        self._ensure_cosine()

    @classmethod
    def from_environment(cls, embedding_model: Embeddings) -> "ChromaDbClient":
        url = (os.getenv("CHROMA_URL") or "").strip() or DEFAULT_URL
        name = (os.getenv("CHROMA_COLLECTION") or "").strip() or DEFAULT_COLLECTION
        return cls(url, name, embedding_model)

    def _ensure_cosine(self) -> None:
        space = (self.collection.metadata or {}).get("hnsw:space")
        if space is None:
            hnsw = (getattr(self.collection, "configuration", None) or {}).get("hnsw") or {}
            space = hnsw.get("space")
        if space != "cosine":
            raise RuntimeError(
                f"Chroma collection '{self.collection.name}' uses distance '{space}', expected 'cosine'"
            )

    @staticmethod
    def _make_id(index: int, text: str, metadata: dict) -> str:
        meta = sorted((str(k), str(v)) for k, v in (metadata or {}).items())
        key = f"{index}\0{meta}\0{text}"
        return str(uuid.uuid5(uuid.NAMESPACE_OID, key))

    def add_segments(self, segments: list[dict]) -> None:
        if not segments:
            return
        texts = [s["text"] for s in segments]
        metadatas = [s.get("metadata") or None for s in segments]
        ids = [self._make_id(i, s["text"], s.get("metadata") or {}) for i, s in enumerate(segments)]
        embeddings = self.embedding_model.embed_documents(texts)
        self.collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

    def search(self, query: str, max_results: int, min_score: float) -> list[dict]:
        embedding = self.embedding_model.embed_query(query)
        res = self.collection.query(
            query_embeddings=[embedding],
            n_results=max_results,
            include=["documents", "metadatas", "distances"],
        )
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        results = []
        for text, meta, dist in zip(docs, metas, dists):
            score = 1.0 - dist  # cosine distance -> similarity
            if score >= min_score:
                results.append({"text": text, "metadata": meta or {}, "score": score})
        return results

    def delete_by_source(self, source_file: str) -> None:
        self.collection.delete(where={"source_file": source_file})
