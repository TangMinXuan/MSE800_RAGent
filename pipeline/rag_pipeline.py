from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from model import Chunk

from pipeline.components.chunk_service import ChunkService
from pipeline.components.prompt_service import PromptService


class RagState(TypedDict, total=False):
    question: str
    history: list[Any]
    rewritten_query: str
    retrieved_chunks: list[Chunk]
    context: str


@dataclass(frozen=True)
class RagResult:
    context: str
    chunks: list[Chunk]


class RagPipeline:
    """
    职责：根据用户问题检索并返回相关知识库资料，由外层Agent结合会话生成答案。

    流程：

    用户输入
      ↓
    ConversationMemory          // 读取历史对话，构建上下文
      ↓
    PromptService.rewrite_query // query改写, 让召回更精准
      ↓
    ChunkService.retrieve       // 召回 top K
      ↓
    ChunkService.buildParentChildChunks
      ↓
    返回检索上下文给外层Agent

    """

    def __init__(self, chunk_service: ChunkService, prompt_service: PromptService):
        self.chunk_service = chunk_service
        self.prompt_service = prompt_service
        self.graph = self._build_graph()

    def _build_graph(self):
        g = StateGraph(RagState)
        g.add_node("rewrite_query", self._rewrite_query)
        g.add_node("retrieve_chunks", self._retrieve_chunks)
        g.add_node("build_context", self._build_context)
        g.add_edge(START, "rewrite_query")
        g.add_edge("rewrite_query", "retrieve_chunks")
        g.add_edge("retrieve_chunks", "build_context")
        g.add_edge("build_context", END)
        return g.compile()

    def _rewrite_query(self, state: RagState) -> dict:
        question = state.get("question")
        if question is None:
            raise RuntimeError("Missing graph state value: question")
        print(f"RAG: raw question: {question}")
        history = [m.to_chat_message() for m in state.get("history", [])]
        return {"rewritten_query": self.prompt_service.rewrite_query(history, question)}

    def _retrieve_chunks(self, state: RagState) -> dict:
        rewritten = state.get("rewritten_query")
        if rewritten is None:
            raise RuntimeError("Missing graph state value: rewrittenQuery")
        print(f"RAG: rewritten query: {rewritten}")
        return {"retrieved_chunks": self.chunk_service.retrieve(rewritten)}

    def _build_context(self, state: RagState) -> dict:
        chunks = state.get("retrieved_chunks")
        if chunks is None:
            raise RuntimeError("Chunks were not retrieved")
        print(f"RAG: building context for retrieved chunks: {len(chunks)}")
        return {"context": self.prompt_service.build_retrieved_context(chunks)}

    def invoke(self, question: str | None, history: list | None) -> RagResult:
        if question is None or not question.strip():
            raise ValueError("Question must not be blank")
        history = [] if history is None else list(history)
        result = self.graph.invoke({"question": question, "history": history})
        context = result.get("context")
        chunks = result.get("retrieved_chunks")
        if context is None:
            raise RuntimeError("RAG graph produced no context")
        if chunks is None:
            raise RuntimeError("RAG graph produced no chunks")
        return RagResult(context=context, chunks=chunks)
