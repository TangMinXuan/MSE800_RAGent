# STUB: replaced in the pipeline migration step
from dataclasses import dataclass

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage


@dataclass(frozen=True)
class TextHistoryMessage:
    role: str
    text: str

    def to_chat_message(self) -> BaseMessage:
        if self.role == "user":
            return HumanMessage(content=self.text)
        if self.role == "ai":
            return AIMessage(content=self.text)
        if self.role == "system":
            return SystemMessage(content=self.text)
        raise ValueError(f"Unknown role: {self.role}")


class ConversationMemory:
    def history_for_tool(self, memory_id: str) -> list[TextHistoryMessage]:
        return []

    def update_conversation_title(self, conversation_id: str, title: str) -> None:
        pass

    def get(self, memory_id: str):
        raise NotImplementedError
