from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from pipeline.components.conversation_memory import ConversationMemory
from pipeline.rag_pipeline import RagPipeline

TANG_FAN_DB_TOOL_NAME = "query_tang_fan_db"
TANG_FAN_DB_COLLECTION = (
    "Query TangFan DB documentation. Use this tool when user asks about TangFan DB "
    "features, configuration, commands, or any TangFan DB specific knowledge."
)

HONEY_TOOL_NAME = "query_maori_honey"
HONEY_TOOL_DESCRIPTION = (
    "Query the private Maori/Manuka honey knowledge base. ALWAYS use this tool first for any "
    "question about honey, Manuka honey, Maori honey, its properties, grading (UMF/MGO), "
    "or industry knowledge. Do not answer such questions without calling it."
)


class _QueryArgs(BaseModel):
    question: str = Field(description="The question to look up in the knowledge base")


class AgentTool:
    def __init__(
        self,
        rag_pipeline: RagPipeline,
        conversation_memory: ConversationMemory,
        name: str = TANG_FAN_DB_TOOL_NAME,
        description: str = TANG_FAN_DB_COLLECTION,
    ):
        self.rag_pipeline = rag_pipeline
        self.conversation_memory = conversation_memory
        self.name = name
        self.description = description

    def search_private_knowledge(self, question: str, conversation_id: str) -> str:
        history = self.conversation_memory.history_for_tool(conversation_id)
        return self.rag_pipeline.invoke(question, history).context

    def as_langchain_tool(self, conversation_id: str) -> StructuredTool:
        def _run(question: str) -> str:
            return self.search_private_knowledge(question, conversation_id)

        return StructuredTool.from_function(
            func=_run,
            name=self.name,
            description=self.description,
            args_schema=_QueryArgs,
        )
