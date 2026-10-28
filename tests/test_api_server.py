import json

from fastapi.testclient import TestClient

from api.api_server import ApiServer, ChatReq, sse


def test_sse_formats_json_and_preserves_maori_characters():
    event = sse("delta", {"content": "Kia ora Māori"})
    assert event.startswith("event: delta\ndata: ")
    assert json.loads(event.split("data: ", 1)[1].strip()) == {"content": "Kia ora Māori"}
    assert event.endswith("\n\n")


def test_health_endpoint_returns_ok():
    client = TestClient(ApiServer().app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_request_schema_accepts_optional_conversation_id():
    req = ChatReq(rawQuery="Hello")
    assert req.conversationId is None
    assert req.rawQuery == "Hello"


def test_stream_without_agent_emits_error_event():
    server = ApiServer()
    events = list(server._stream(ChatReq(rawQuery="Hello")))
    assert len(events) == 1
    assert "event: error" in events[0]
    assert json.loads(events[0].split("data: ", 1)[1]) == {"message": "Agent not wired yet"}


def test_stream_new_conversation_emits_id_deltas_title_and_done():
    class Agent:
        def chat_stream(self, conversation_id, query):
            assert query == "Hello"
            yield "Hi "
            yield "there"

    class Memory:
        def get(self, conversation_id):
            return self

        def messages(self):
            return []

        def update_conversation_title(self, conversation_id, title):
            self.updated = (conversation_id, title)

    class ChatService:
        def generate_title(self, messages):
            return "Greeting"

    server = ApiServer(Agent(), Memory(), ChatService())
    events = list(server._stream(ChatReq(rawQuery="Hello")))
    names = [event.split("\n", 1)[0] for event in events]
    assert names[0] == "event: conversationId"
    assert names[1:3] == ["event: delta", "event: delta"]
    assert names[3] == "event: conversationTitle"
    assert names[4] == "event: done"
    conversation_id = json.loads(events[0].split("data: ", 1)[1])["conversationId"]
    assert conversation_id
    assert json.loads(events[3].split("data: ", 1)[1]) == {"conversationId": conversation_id, "title": "Greeting"}


def test_stream_existing_conversation_does_not_generate_title():
    class Agent:
        def chat_stream(self, conversation_id, query):
            yield "answer"

    server = ApiServer(agent=Agent())
    events = list(server._stream(ChatReq(conversationId="existing", rawQuery="Hello")))
    assert not any("event: conversationId" in event for event in events)
    assert not any("event: conversationTitle" in event for event in events)
    assert any("event: done" in event for event in events)


def test_chat_post_endpoint_currently_returns_placeholder_response():
    """Characterizes current code: endpoint is a stub and does not call the agent."""
    client = TestClient(ApiServer().app)
    response = client.post("/api/chat", json={"rawQuery": "What is TangFan DB?"})
    assert response.status_code == 200
    body = response.json()
    assert body["llmResp"] == "1+1=2"
    assert body["conversationId"] == "blanket for test"
