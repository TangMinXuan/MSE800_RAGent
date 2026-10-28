from types import SimpleNamespace

from agent_tool import AgentTool, HONEY_TOOL_NAME, TANG_FAN_DB_TOOL_NAME


class FakeMemory:
    def history_for_tool(self, conversation_id):
        return ["prior history", conversation_id]


class FakePipeline:
    def __init__(self):
        self.calls = []

    def invoke(self, question, history):
        self.calls.append((question, history))
        return SimpleNamespace(context="retrieved context")


def test_search_private_knowledge_passes_question_and_history():
    pipeline, memory = FakePipeline(), FakeMemory()
    tool = AgentTool(pipeline, memory)
    assert tool.search_private_knowledge("question", "c1") == "retrieved context"
    assert pipeline.calls == [("question", ["prior history", "c1"])]


def test_as_langchain_tool_exposes_name_and_invokes_pipeline():
    pipeline, memory = FakePipeline(), FakeMemory()
    tool = AgentTool(pipeline, memory, name=HONEY_TOOL_NAME, description="honey docs")
    langchain_tool = tool.as_langchain_tool("c1")
    assert langchain_tool.name == HONEY_TOOL_NAME
    assert langchain_tool.description == "honey docs"
    assert langchain_tool.invoke({"question": "What is MGO?"}) == "retrieved context"
    assert pipeline.calls[0][0] == "What is MGO?"


def test_default_tool_name_is_tangfan_db():
    tool = AgentTool(FakePipeline(), FakeMemory())
    assert tool.name == TANG_FAN_DB_TOOL_NAME
