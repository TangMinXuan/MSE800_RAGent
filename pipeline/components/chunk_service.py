from langchain_text_splitters import RecursiveCharacterTextSplitter

from pipeline.components.vector_store_service import VectorStoreService
from model import Chunk


class ChunkService:
    def __init__(self, vector_store_service: VectorStoreService):
        self.vector_store_service = vector_store_service

    def split(self, document_text: str, source_file: str | None) -> list[Chunk]:
        if source_file is None:
            source_file = "unknown"
        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
        return [
            Chunk(id=i, content=text, source_file=source_file)
            for i, text in enumerate(splitter.split_text(document_text))
        ]

    def retrieve(self, rewritten_query: str) -> list[Chunk]:
        return self.vector_store_service.retrieve(rewritten_query, 3)
