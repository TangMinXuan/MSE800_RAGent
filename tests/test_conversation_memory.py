import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from pipeline.components.conversation_memory import (
    ChatMemory,
    ConversationMemory,
    TextHistoryMessage,
    _InMemoryStore,
    _messages_from_json,
    _messages_to_json,
)


def test_text_history_message_converts_supported_roles():
    assert isinstance(TextHistoryMessage("user", "hello").to_chat_message(), HumanMessage)
    assert isinstance(TextHistoryMessage("assistant", "hello").to_chat_message(), AIMessage)
    with pytest.raises(ValueError, match="Unsupported history role"):
        TextHistoryMessage("system", "nope").to_chat_message()


def test_chat_memory_adds_reads_clears_and_keeps_recent_messages():
    memory = ChatMemory("c1", max_messages=3)
    memory.add(HumanMessage(content="one"))
    memory.add(AIMessage(content="two"))
    memory.add(HumanMessage(content="three"))
    memory.add(AIMessage(content="four"))
    assert [m.content for m in memory.messages()] == ["two", "three", "four"]
    memory.clear()
    assert memory.messages() == []


def test_chat_memory_keeps_single_system_message_at_front():
    memory = ChatMemory("c1")
    memory.add(SystemMessage(content="first system"))
    memory.add(HumanMessage(content="question"))
    memory.add(SystemMessage(content="replacement system"))
    messages = memory.messages()
    assert len([m for m in messages if isinstance(m, SystemMessage)]) == 1
    assert messages[0].content == "replacement system"


def test_chat_memory_trimming_drops_orphaned_tool_messages():
    memory = ChatMemory("c1", max_messages=2)
    memory.add(HumanMessage(content="q1"))
    memory.add(AIMessage(content="a1"))
    memory.add(ToolMessage(content="tool result", tool_call_id="call1"))
    memory.add(HumanMessage(content="q2"))
    assert not (memory.messages() and isinstance(memory.messages()[0], ToolMessage))


def test_conversation_memory_reuses_in_memory_instance_and_rejects_none():
    memory = ConversationMemory()
    assert memory.get_from_memory("c1") is memory.get_from_memory("c1")
    with pytest.raises(ValueError, match="must not be null"):
        memory.get_from_memory(None)
    with pytest.raises(ValueError, match="must not be null"):
        memory.get(None)


def test_history_for_tool_excludes_latest_user_question_and_ignores_tool_messages(monkeypatch):
    memory = ConversationMemory()
    chat_memory = ChatMemory("c1")
    chat_memory.add(HumanMessage(content="previous question"))
    chat_memory.add(AIMessage(content="previous answer"))
    chat_memory.add(HumanMessage(content="current question"))
    monkeypatch.setattr(memory, "get", lambda _id: chat_memory)
    result = memory.history_for_tool("c1")
    assert [(item.role, item.text) for item in result] == [
        ("user", "previous question"), ("assistant", "previous answer")
    ]


def test_message_serialization_round_trip():
    messages = [HumanMessage(content="Kia ora"), AIMessage(content="Hello")]
    assert [m.content for m in _messages_from_json(_messages_to_json(messages))] == ["Kia ora", "Hello"]


def test_in_memory_store_isolates_copies():
    store = _InMemoryStore()
    store.update_messages("c1", [HumanMessage(content="hello")])
    result = store.get_messages("c1")
    result.clear()
    assert len(store.get_messages("c1")) == 1
