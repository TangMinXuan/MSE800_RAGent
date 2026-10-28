import pytest
from langchain_core.messages import AIMessage, HumanMessage

from pipeline.components.conversation_memory import _PostgresChatMemoryStore


class FakeDb:
    def __init__(self, rows=None, fail=None):
        self.rows = rows or []
        self.fail = fail
        self.calls = []

    def select(self, *args):
        self.calls.append(("select", args))
        if self.fail == "select":
            raise RuntimeError("database down")
        return self.rows

    def insert(self, *args):
        self.calls.append(("insert", args))
        if self.fail == "insert":
            raise RuntimeError("database down")

    def update(self, *args):
        self.calls.append(("update", args))
        if self.fail == "update":
            raise RuntimeError("database down")

    def delete(self, *args):
        self.calls.append(("delete", args))
        if self.fail == "delete":
            raise RuntimeError("database down")


def test_get_messages_returns_empty_when_no_rows_or_no_history():
    assert _PostgresChatMemoryStore(FakeDb()).get_messages("c1") == []
    row = type("Row", (), {"history": None})()
    assert _PostgresChatMemoryStore(FakeDb([row])).get_messages("c1") == []


def test_get_messages_deserializes_history():
    db = FakeDb()
    store = _PostgresChatMemoryStore(db)
    messages = [HumanMessage(content="question"), AIMessage(content="answer")]
    from pipeline.components.conversation_memory import _messages_to_json
    row = type("Row", (), {"history": _messages_to_json(messages)})()
    db.rows = [row]
    assert [m.content for m in store.get_messages("c1")] == ["question", "answer"]


def test_update_messages_inserts_new_conversation_or_updates_existing():
    db = FakeDb()
    store = _PostgresChatMemoryStore(db)
    store.update_messages("c1", [HumanMessage(content="hello")])
    assert [call[0] for call in db.calls] == ["select", "insert"]
    db.calls.clear()
    db.rows = [object()]
    store.update_messages("c1", [AIMessage(content="answer")])
    assert [call[0] for call in db.calls] == ["select", "update"]


def test_delete_messages_delegates_to_database():
    db = FakeDb()
    _PostgresChatMemoryStore(db).delete_messages("c1")
    assert db.calls[0][0] == "delete"


@pytest.mark.parametrize(("method", "expected"), [
    ("get_messages", "Failed to load conversation history"),
    ("update_messages", "Failed to save conversation history"),
    ("delete_messages", "Failed to delete conversation history"),
])
def test_database_errors_are_wrapped(method, expected):
    db = FakeDb(fail="select" if method in ("get_messages", "update_messages") else "delete")
    store = _PostgresChatMemoryStore(db)
    with pytest.raises(RuntimeError, match=expected):
        if method == "get_messages":
            store.get_messages("c1")
        elif method == "update_messages":
            store.update_messages("c1", [HumanMessage(content="hello")])
        else:
            store.delete_messages("c1")
