from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from model import Chunk

from aiagent.pipeline.components.chunk_service import ChunkService
from aiagent.pipeline.components.document_loader import DocumentLoader
from aiagent.pipeline.components.vector_store_service import VectorStoreService


class IngestionState(TypedDict, total=False):
    file: Path
    document_text: str
    source_file: str
    chunks: list[Chunk]
    ingested_count: int


class IngestionPipeline:
    def __init__(
        self,
        document_loader: DocumentLoader,
        chunk_service: ChunkService,
        vector_store_service: VectorStoreService,
    ):
        self.document_loader = document_loader
        self.chunk_service = chunk_service
        self.vector_store_service = vector_store_service
        self.graph = self._build_graph()

    def _build_graph(self):
        g = StateGraph(IngestionState)
        g.add_node("load_document", self._load_document)
        g.add_node("split_document", self._split_document)
        g.add_node("store_chunks", self._store_chunks)
        g.add_edge(START, "load_document")
        g.add_edge("load_document", "split_document")
        g.add_edge("split_document", "store_chunks")
        g.add_edge("store_chunks", END)
        return g.compile()

    def _load_document(self, state: IngestionState) -> dict:
        file = state.get("file")
        if file is None:
            raise ValueError("A document file is required")
        document = self.document_loader.load_document(file)
        return {"document_text": document.text, "source_file": str(Path(file).resolve())}

    def _split_document(self, state: IngestionState) -> dict:
        return {"chunks": self.chunk_service.split(state["document_text"], state["source_file"])}

    def _store_chunks(self, state: IngestionState) -> dict:
        chunks = state["chunks"]
        self.vector_store_service.delete_by_source(state["source_file"])
        self.vector_store_service.upsert(chunks)
        return {"ingested_count": len(chunks)}

    def ingest(self, file: Path | None) -> int:
        if file is None or not Path(file).is_file():
            raise ValueError(f"Document file does not exist: {file}")
        result = self.graph.invoke({"file": Path(file)})
        return result["ingested_count"]


def main() -> None:
    # TODO: wire in the next step, in this order:
    #   embedding model -> ChromaDbClient.from_environment -> VectorStoreService
    #   -> ChunkService -> IngestionPipeline
    #   ingest src/main/resources/base.txt and print f"已处理文档块数: {count}"
    raise NotImplementedError


if __name__ == "__main__":
    main()
