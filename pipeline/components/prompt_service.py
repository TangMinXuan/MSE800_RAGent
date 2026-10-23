from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from infra.llm.chat_service import ChatService
from model import Chunk
from util.string_util import is_blank

REWRITE_PROMPT_TEMPLATE = """Rewrite the latest user question as a concise, standalone search query for document retrieval.
    Rules:
    1. If the subject of the question is uncertain, it is presumed to be TangFan DB
    2. If the question is already clear and specific, return it unchanged
    3. Return only the rewritten query, no explanation
Return only the rewritten question:
{originalQuery}
"""

REWRITE_SYSTEM_PROMPT = """Rewrite the latest question in the conversation as a standalone question
suitable for document retrieval. Use the previous conversation to resolve
references when helpful. Return only the rewritten question; do not answer it.
"""


# chunks + 历史 + 问题 → 组装Prompt
class PromptService:
    def __init__(self, chat_service: ChatService):
        self.chat_service = chat_service

    def build_retrieved_context(self, retrieved_chunks: list[Chunk]) -> str:
        context = ""
        for chunk in retrieved_chunks:
            if not is_blank(chunk.content):
                context += f"[Source: {chunk.source_file}]\n{chunk.content}\n\n"
        if not context:
            return "No relevant TangFan DB documentation was found."
        return "[Retrieved TangFan DB reference material]\n" + context

    def rewrite_query(self, history: list[BaseMessage], original_query: str) -> str:
        prompt = REWRITE_PROMPT_TEMPLATE.replace("{originalQuery}", original_query)
        if not history:
            return self.chat_service.chat(prompt)
        messages = [SystemMessage(content=REWRITE_SYSTEM_PROMPT), *history, HumanMessage(content=prompt)]
        return self.chat_service.chat(messages)
