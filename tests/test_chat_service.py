from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from infra.llm.chat_service import ChatService, TITLE_REQUEST, TITLE_SYSTEM_PROMPT


class FakeModel:
    def __init__(self, content="answer"):
        self.content = content
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content=self.content)


def test_chat_wraps_string_prompt_in_human_message():
    model = FakeModel("hello")
    assert ChatService(model).chat("say hello") == "hello"
    assert len(model.calls) == 1
    assert isinstance(model.calls[0][0], HumanMessage)
    assert model.calls[0][0].content == "say hello"


def test_chat_passes_message_list_unchanged():
    model = FakeModel("response")
    messages = [SystemMessage(content="system"), HumanMessage(content="question")]
    assert ChatService(model).chat(messages) == "response"
    assert model.calls[0] == messages


def test_generate_title_adds_title_instructions_and_strips_output():
    model = FakeModel("  A short title  \n")
    history = [HumanMessage(content="How do I configure TangFan?")]
    assert ChatService(model).generate_title(history) == "A short title"
    messages = model.calls[0]
    assert messages[0].content == TITLE_SYSTEM_PROMPT
    assert messages[-1].content == TITLE_REQUEST
    assert messages[1] == history[0]
