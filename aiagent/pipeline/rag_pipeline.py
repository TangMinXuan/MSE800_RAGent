from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from model import Chunk

from aiagent.pipeline.components.chunk_service import ChunkService
from aiagent.pipeline.components.prompt_service import PromptService


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
    """RAG 流程：
    1. rewrite_query：结合对话历史改写用户问题
    2. retrieve_chunks：用改写后的问题检索文档块
    3. build_context：把检索到的文档块拼成上下文
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
            raise RuntimeError("question is missing from state")
        print(f"RAG: raw question: {question}")
        history = [m.to_chat_message() for m in state.get("history", [])]
        return {"rewritten_query": self.prompt_service.rewrite_query(history, question)}

    def _retrieve_chunks(self, state: RagState) -> dict:
        rewritten = state["rewritten_query"]
        print(f"RAG: rewritten query: {rewritten}")
        return {"retrieved_chunks": self.chunk_service.retrieve(rewritten)}

    def _build_context(self, state: RagState) -> dict:
        chunks = state["retrieved_chunks"]
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
            raise RuntimeError("RAG pipeline did not produce a context")
        if chunks is None:
            raise RuntimeError("RAG pipeline did not produce retrieved chunks")
        return RagResult(context=context, chunks=chunks)
