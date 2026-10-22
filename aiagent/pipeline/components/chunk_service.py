# STUB: replaced in the pipeline migration step
from model import Chunk

from aiagent.pipeline.components.vector_store_service import VectorStoreService


class ChunkService:
    def __init__(self, vector_store_service: VectorStoreService):
        self.vector_store_service = vector_store_service

    def split(self, document_text: str, source_file: str) -> list[Chunk]:
        raise NotImplementedError

    def retrieve(self, rewritten_query: str) -> list[Chunk]:
        raise NotImplementedError
