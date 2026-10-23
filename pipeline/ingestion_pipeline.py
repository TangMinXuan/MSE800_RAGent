from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from pypdf import PdfReader

from model import Chunk

from pipeline.components.chunk_service import ChunkService
from pipeline.components.vector_store_service import VectorStoreService


class IngestionState(TypedDict, total=False):
    file: Path
    document_text: str
    source_file: str
    chunks: list[Chunk]
    ingested_count: int


class IngestionPipeline:
    """
    职责：把原始文档 → 变成 Vector DB 里的chunks
    流程：
    原始文件（PDF/MD/TXT）
      ↓
    load_document          // 读取文件，转成纯文本
      ↓
    ChunkSplitter          // 按语义/段落切割成小块
      ↓
    EmbeddingService       // 调用Embedding模型，文本→向量
      ↓
    VectorStoreClient      // 存入Chroma

    触发时机：
      - 手动触发（新文档上传时）
      - 不在对话流程中
      - 一次性或定期批量运行
    """

    def __init__(
        self,
        chunk_service: ChunkService,
        vector_store_service: VectorStoreService,
    ):
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
        file = Path(file)
        if not file.is_file():
            raise FileNotFoundError(f"Document file not found: {file}")
        if file.suffix.lower() == ".pdf":
            reader = PdfReader(str(file))
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            try:
                text = file.read_text(encoding="utf-8")
            except UnicodeDecodeError as exception:
                raise ValueError(f"Unsupported document type: {file.suffix or file.name}") from exception
        return {"document_text": text, "source_file": str(file.resolve())}

    def _split_document(self, state: IngestionState) -> dict:
        document_text = state.get("document_text")
        if document_text is None:
            raise RuntimeError("The document was not loaded")
        source_file = state.get("source_file")
        if source_file is None:
            raise RuntimeError("The source file was not retained")
        return {"chunks": self.chunk_service.split(document_text, source_file)}

    def _store_chunks(self, state: IngestionState) -> dict:
        chunks = state.get("chunks")
        if chunks is None:
            raise RuntimeError("The document was not split")
        source_file = state.get("source_file")
        if source_file is None:
            raise RuntimeError("The source file was not retained")
        self.vector_store_service.delete_by_source(source_file)
        self.vector_store_service.upsert(chunks)
        return {"ingested_count": len(chunks)}

    def ingest(self, file: Path | None) -> int:
        if file is None or not Path(file).is_file():
            raise ValueError(f"Document file does not exist: {file}")
        result = self.graph.invoke({"file": Path(file)})
        if not result:
            raise RuntimeError("Document ingestion produced no state")
        count = result.get("ingested_count")
        if count is None:
            raise RuntimeError("Document ingestion produced no chunk count")
        return count


def main() -> None:
    from langchain_huggingface import HuggingFaceEmbeddings

    from infra.database.chroma_db_client import ChromaDbClient

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    chroma_db_client = ChromaDbClient.from_environment(embedding_model)
    vector_store_service = VectorStoreService(chroma_db_client)
    chunk_service = ChunkService(vector_store_service)

    ingestion_pipeline = IngestionPipeline(chunk_service, vector_store_service)
    # ingested_chunks = ingestion_pipeline.ingest(Path("/Users/tangfan/PycharmProjects/RAGent/resources/base.txt"))
    ingested_chunks = ingestion_pipeline.ingest(Path("/Users/tangfan/PycharmProjects/RAGent/resources/maori_honey_description.txt"))
    print(f"已处理文档块数: {ingested_chunks}")


if __name__ == "__main__":
    main()
