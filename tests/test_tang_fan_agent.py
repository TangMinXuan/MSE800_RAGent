from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from tang_fan_agent import TangFanAgent, _text_of


def test_text_of_handles_string_and_content_blocks():
    assert _text_of("hello") == "hello"
    assert _text_of([{"type": "text", "text": "Kia "}, {"type": "text", "text": "ora"}]) == "Kia ora"
    assert _text_of([{"type": "image", "url": "x"}]) == ""
    assert _text_of(None) == ""


def test_start_adds_user_message_and_returns_prior_history():
    class Memory:
        def __init__(self):
            self.items = {"c1": [AIMessage(content="previous answer")]}
        def get(self, key):
            return self
        def messages(self):
            return list(self.items["c1"])
        def add(self, message):
            self.items["c1"].append(message)

    memory = Memory()
    agent = TangFanAgent(None, [], memory)
    returned_memory, inputs = agent._start("c1", "new question")
    assert returned_memory is memory
    assert [m.content for m in inputs] == ["previous answer", "new question"]
    assert isinstance(inputs[-1], HumanMessage)


def test_save_new_messages_only_appends_messages_after_input():
    class Memory:
        def __init__(self): self.added = []
        def add(self, message): self.added.append(message)
    memory = Memory()
    existing = HumanMessage(content="question")
    answer = AIMessage(content="answer")
    TangFanAgent._save_new_messages(memory, {"messages": [existing, answer]}, 1)
    assert memory.added == [answer]


def test_chat_uses_final_ai_message_and_saves_new_messages(monkeypatch):
    class Memory:
        def __init__(self): self.items = []
        def get(self, key): return self
        def messages(self): return list(self.items)
        def add(self, message): self.items.append(message)
    memory = Memory()
    agent = TangFanAgent(None, [], memory)
    class FakeGraph:
        def invoke(self, payload):
            return {"messages": [*payload["messages"], AIMessage(content="answer")]} 
    monkeypatch.setattr(agent, "_agent", lambda conversation_id: FakeGraph())
    assert agent.chat("c1", "question") == "answer"
    assert [m.content for m in memory.items] == ["question", "answer"]


def test_chat_stream_yields_text_chunks_and_saves_final_state(monkeypatch):
    class Memory:
        def __init__(self): self.items = []
        def get(self, key): return self
        def messages(self): return list(self.items)
        def add(self, message): self.items.append(message)
    memory = Memory()
    agent = TangFanAgent(None, [], memory)
    class FakeGraph:
        def stream(self, payload, stream_mode):
            yield "messages", (AIMessageChunk(content="Hello "), {})
            yield "messages", (AIMessageChunk(content="world"), {})
            yield "values", {"messages": [*payload["messages"], AIMessage(content="Hello world")]}
    monkeypatch.setattr(agent, "_agent", lambda conversation_id: FakeGraph())
    assert list(agent.chat_stream("c1", "question")) == ["Hello ", "world"]
    assert [m.content for m in memory.items] == ["question", "Hello world"]
