from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from pipeline.components.conversation_memory import ConversationMemory
from pipeline.rag_pipeline import RagPipeline

TOOL_NAME = "query_tang_fan_db"
TOOL_DESCRIPTION = (
    "Query TangFan DB documentation. Use this tool when user asks about TangFan DB "
    "features, configuration, commands, or any TangFan DB specific knowledge."
)


class _QueryArgs(BaseModel):
    question: str = Field(description="The question to look up in TangFan DB documentation")


class TangFanDbTool:
    def __init__(self, rag_pipeline: RagPipeline, conversation_memory: ConversationMemory):
        self.rag_pipeline = rag_pipeline
        self.conversation_memory = conversation_memory

    def query_tang_fan_db(self, question: str, conversation_id: str) -> str:
        history = self.conversation_memory.history_for_tool(conversation_id)
        return self.rag_pipeline.invoke(question, history).context

    def as_langchain_tool(self, conversation_id: str) -> StructuredTool:
        def _run(question: str) -> str:
            return self.query_tang_fan_db(question, conversation_id)

        return StructuredTool.from_function(
            func=_run,
            name=TOOL_NAME,
            description=TOOL_DESCRIPTION,
            args_schema=_QueryArgs,
        )
