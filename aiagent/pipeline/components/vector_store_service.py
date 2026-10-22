# STUB: replaced in the pipeline migration step
from typing import Any

from model import Chunk


class VectorStoreService:
    def __init__(self, chroma_db_client: Any):
        self.chroma_db_client = chroma_db_client

    def upsert(self, chunks: list[Chunk]) -> None:
        raise NotImplementedError

    def retrieve(self, query: str, top_k: int) -> list[Chunk]:
        raise NotImplementedError

    def delete_by_source(self, source_file: str) -> None:
        raise NotImplementedError
