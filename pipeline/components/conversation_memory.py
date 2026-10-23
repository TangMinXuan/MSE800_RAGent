"""会话记忆（对应 Java 的 ChatMemoryProvider / MessageWindowChatMemory / ChatMemoryStore）。

Agent 应通过 ``memory.get(conversation_id)`` 获得 ``ChatMemory``，
使用 ``messages()`` / ``add()`` 读写历史；后续装配步骤再把 TangFanAgent 切换过来。
"""
import json
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    messages_from_dict,
    messages_to_dict,
)

from infra.database.postgres_db_client import PostgresDbClient
from model import ConversationHistory
from util.string_util import is_blank

MAX_MESSAGES = 20
CONVERSATION_HISTORY_TABLE = "conversation_history"


@dataclass(frozen=True)
class TextHistoryMessage:
    role: str
    text: str

    def to_chat_message(self) -> BaseMessage:
        if self.role == "user":
            return HumanMessage(content=self.text)
        if self.role == "assistant":
            return AIMessage(content=self.text)
        raise ValueError(f"Unsupported history role: {self.role}")


def _messages_to_json(messages: list[BaseMessage]) -> str:
    return json.dumps(messages_to_dict(messages), ensure_ascii=False)


def _messages_from_json(s: str) -> list[BaseMessage]:
    return messages_from_dict(json.loads(s))


class _PostgresChatMemoryStore:
    def __init__(self, db_client: PostgresDbClient):
        self.db_client = db_client

    def get_messages(self, id: Any) -> list[BaseMessage]:
        try:
            rows = self.db_client.select(
                CONVERSATION_HISTORY_TABLE,
                ConversationHistory,
                ConversationHistory(conversation_id=str(id)),
            )
            if not rows or rows[0].history is None:
                return []
            return _messages_from_json(rows[0].history)
        except Exception as exception:
            print(exception)
            raise RuntimeError("Failed to load conversation history from PostgreSQL") from exception

    def update_messages(self, id: Any, messages: list[BaseMessage]) -> None:
        serialized = _messages_to_json(messages)
        values = ConversationHistory(conversation_id=str(id), history=serialized)
        conditions = ConversationHistory(conversation_id=str(id))
        try:
            rows = self.db_client.select(CONVERSATION_HISTORY_TABLE, ConversationHistory, conditions)
            if not rows:
                self.db_client.insert(CONVERSATION_HISTORY_TABLE, values)
            else:
                self.db_client.update(CONVERSATION_HISTORY_TABLE, values, conditions)
        except Exception as exception:
            raise RuntimeError("Failed to save conversation history to PostgreSQL") from exception

    def delete_messages(self, id: Any) -> None:
        try:
            self.db_client.delete(
                CONVERSATION_HISTORY_TABLE, ConversationHistory(conversation_id=str(id))
            )
        except Exception as exception:
            raise RuntimeError("Failed to delete conversation history from PostgreSQL") from exception


class _InMemoryStore:
    def __init__(self):
        self._messages: list[BaseMessage] = []

    def get_messages(self, id: Any) -> list[BaseMessage]:
        return list(self._messages)

    def update_messages(self, id: Any, messages: list[BaseMessage]) -> None:
        self._messages = list(messages)

    def delete_messages(self, id: Any) -> None:
        self._messages = []


class ChatMemory:
    """对应 MessageWindowChatMemory：保留最近 MAX_MESSAGES 条消息。"""

    def __init__(self, id: Any, store: Any = None, max_messages: int = MAX_MESSAGES):
        self.id = id
        self.max_messages = max_messages
        self._store = store if store is not None else _InMemoryStore()

    def messages(self) -> list[BaseMessage]:
        return list(self._store.get_messages(self.id))

    def add(self, message: BaseMessage) -> None:
        messages = self.messages()
        if isinstance(message, SystemMessage):
            messages = [m for m in messages if not isinstance(m, SystemMessage)]
            messages.insert(0, message)
        else:
            messages.append(message)
        self._store.update_messages(self.id, self._trim(messages))

    def clear(self) -> None:
        self._store.delete_messages(self.id)

    def _trim(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        system = [messages[0]] if messages and isinstance(messages[0], SystemMessage) else []
        rest = messages[len(system):]
        rest = rest[max(0, len(rest) - (self.max_messages - len(system))):]
        # 丢弃因裁剪而失去对应 tool call 的 ToolMessage
        while rest and isinstance(rest[0], ToolMessage):
            rest.pop(0)
        return system + rest


class ConversationMemory:
    def __init__(self):
        self._memories: dict[Any, ChatMemory] = {}

    def get_from_memory(self, memory_id: Any) -> ChatMemory:
        if memory_id is None:
            raise ValueError("Conversation ID must not be null")
        return self._memories.setdefault(memory_id, ChatMemory(id=memory_id))

    def get(self, memory_id: Any) -> ChatMemory:
        if memory_id is None:
            raise ValueError("Conversation ID must not be null")
        store = _PostgresChatMemoryStore(PostgresDbClient.from_environment())
        return ChatMemory(id=str(memory_id), store=store)

    def history_for_tool(self, memory_id: Any) -> list[TextHistoryMessage]:
        messages = list(self.get(memory_id).messages())
        # 最后一条 HumanMessage 是当前正在处理的用户问题，不进入历史
        for index in range(len(messages) - 1, -1, -1):
            if isinstance(messages[index], HumanMessage):
                del messages[index]
                break
        result: list[TextHistoryMessage] = []
        for message in messages:
            result.extend(self._to_text_history_message(message))
        return result

    def update_conversation_title(self, conversation_id: str, title: str) -> None:
        try:
            PostgresDbClient.from_environment().update(
                CONVERSATION_HISTORY_TABLE,
                ConversationHistory(conversation_title=title),
                ConversationHistory(conversation_id=conversation_id),
            )
        except Exception as exception:
            raise RuntimeError("Failed to save conversation title to PostgreSQL") from exception

    @staticmethod
    def _to_text_history_message(message: BaseMessage) -> list[TextHistoryMessage]:
        if isinstance(message, HumanMessage) and isinstance(message.content, str):
            return [TextHistoryMessage("user", message.content)]
        if (
            isinstance(message, AIMessage)
            and not message.tool_calls
            and isinstance(message.content, str)
            and not is_blank(message.content)
        ):
            return [TextHistoryMessage("assistant", message.content)]
        return []
