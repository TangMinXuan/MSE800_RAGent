from langchain_core.messages import HumanMessage, SystemMessage

from model import Chunk
from pipeline.components.prompt_service import PromptService, REWRITE_SYSTEM_PROMPT


class FakeChatService:
    def __init__(self, result="rewritten query"):
        self.result = result
        self.calls = []

    def chat(self, prompt):
        self.calls.append(prompt)
        return self.result


def test_build_context_formats_chunks_and_sources():
    service = PromptService(FakeChatService())
    context = service.build_retrieved_context([
        Chunk(0, "TangFan supports SQL", "base.txt"),
        Chunk(1, "   ", "empty.txt"),
        Chunk(2, "Use a configuration file", "config.md"),
    ])
    assert context.startswith("[Retrieved TangFan DB reference material]")
    assert "[Source: base.txt]\nTangFan supports SQL" in context
    assert "empty.txt" not in context
    assert "[Source: config.md]" in context


def test_build_context_returns_no_results_message_for_empty_or_blank_chunks():
    service = PromptService(FakeChatService())
    assert service.build_retrieved_context([]) == "No relevant TangFan DB documentation was found."
    assert service.build_retrieved_context([Chunk(0, "  ", "empty.txt")]) == "No relevant TangFan DB documentation was found."


def test_rewrite_query_without_history_sends_string_prompt():
    chat = FakeChatService("clean query")
    assert PromptService(chat).rewrite_query([], "how do I configure it?") == "clean query"
    assert isinstance(chat.calls[0], str)
    assert "how do I configure it?" in chat.calls[0]


def test_rewrite_query_with_history_sends_messages():
    chat = FakeChatService("resolved query")
    history = [HumanMessage(content="I mean TangFan DB")]
    assert PromptService(chat).rewrite_query(history, "how to configure it?") == "resolved query"
    messages = chat.calls[0]
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == REWRITE_SYSTEM_PROMPT
    assert messages[1].content == "I mean TangFan DB"
    assert isinstance(messages[-1], HumanMessage)
