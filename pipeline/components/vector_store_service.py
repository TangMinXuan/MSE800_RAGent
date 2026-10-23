from infra.database.chroma_db_client import ChromaDbClient
from model import Chunk
from util.string_util import is_blank

CHUNK_ID = "chunk_id"
SOURCE_FILE = "source_file"


class VectorStoreService:
    def __init__(self, chroma_db_client: ChromaDbClient):
        self.chroma_db_client = chroma_db_client

    # 写入/覆盖
    def upsert(self, chunk_list: list[Chunk]) -> None:
        segments = [
            {
                "text": chunk.content,
                "metadata": {CHUNK_ID: chunk.id, SOURCE_FILE: chunk.source_file},
            }
            for chunk in chunk_list
        ]
        self.chroma_db_client.add_segments(segments)

    # 召回
    def retrieve(self, query: str, top_k: int) -> list[Chunk]:
        matches = self.chroma_db_client.search(query, top_k, 0.3)
        return [
            Chunk(
                id=int(match["metadata"][CHUNK_ID]),
                content=match["text"],
                source_file=match["metadata"][SOURCE_FILE],
            )
            for match in matches
        ]

    # 按文档删全部chunks
    def delete_by_source(self, source_file: str | None) -> None:
        if is_blank(source_file):
            raise ValueError("Source file must not be blank")
        self.chroma_db_client.delete_by_source(source_file)
