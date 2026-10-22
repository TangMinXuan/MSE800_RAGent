# STUB: replaced in the pipeline migration step
from typing import Any

from langchain_core.messages import BaseMessage

from model import Chunk


class PromptService:
    def __init__(self, chat_service: Any):
        self.chat_service = chat_service

    def build_retrieved_context(self, chunks: list[Chunk]) -> str:
        raise NotImplementedError

    def rewrite_query(self, history: list[BaseMessage], original_query: str) -> str:
        raise NotImplementedError
