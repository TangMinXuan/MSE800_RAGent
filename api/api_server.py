import json
import traceback
import uuid
from typing import Any, Iterator

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from util.string_util import is_blank

PORT = 7070


class ChatReq(BaseModel):
    conversationId: str | None = None
    rawQuery: str


class ChatResp(BaseModel):
    code: int
    msg: str
    conversationId: str | None = None
    conversationTitle: str | None = None
    llmResp: str | None = None


def sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class ApiServer:
    def __init__(self, agent: Any = None, conversation_memory: Any = None, chat_service: Any = None):
        self.agent = agent
        self.conversation_memory = conversation_memory
        self.chat_service = chat_service
        self.app = self._build_app()

    def initialize_agent(self) -> None:
        """装配 Agent（对应 Java 的 initializeAgent）；依赖在此处才导入/创建。"""
        import os

        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_openai import ChatOpenAI

        from infra.database.chroma_db_client import ChromaDbClient
        from infra.llm.chat_service import ChatService
        from pipeline.components.chunk_service import ChunkService
        from pipeline.components.conversation_memory import ConversationMemory
        from pipeline.components.prompt_service import PromptService
        from pipeline.components.vector_store_service import VectorStoreService
        from pipeline.rag_pipeline import RagPipeline
        from tang_fan_agent import TangFanAgent
        from tang_fan_db_tool import TangFanDbTool

        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        chroma_db_client = ChromaDbClient.from_environment(embedding_model)
        vector_store_service = VectorStoreService(chroma_db_client)
        chunk_service = ChunkService(vector_store_service)

        self.chat_service = ChatService()
        rag_pipeline = RagPipeline(chunk_service, PromptService(self.chat_service))
        self.conversation_memory = ConversationMemory()
        tool = TangFanDbTool(rag_pipeline, self.conversation_memory)

        agent_model = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("API_KEY"),
            model="openai/gpt-4o-mini",
            streaming=True,
        )
        self.agent = TangFanAgent(agent_model, tool, self.conversation_memory)

    def _build_app(self) -> FastAPI:
        app = FastAPI()

        @app.get("/health")
        def health():
            return {"status": "ok"}

        @app.post("/api/chat", response_model=ChatResp)
        def chat(req: ChatReq):
            print(req)
            return ChatResp(
                code=200,
                msg="success",
                conversationId="blanket for test",
                conversationTitle="test",
                llmResp="1+1=2",
            )

        @app.post("/api/chat/stream")
        def chat_stream(req: ChatReq):
            print(req)
            headers = {
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
            return StreamingResponse(
                self._stream(req), media_type="text/event-stream", headers=headers
            )

        return app

    def _stream(self, req: ChatReq) -> Iterator[str]:
        if self.agent is None:
            yield sse("error", {"message": "Agent not wired yet"})
            return

        new_conversation = is_blank(req.conversationId)
        conversation_id = str(uuid.uuid4()) if new_conversation else req.conversationId
        if new_conversation:
            yield sse("conversationId", {"conversationId": conversation_id})

        try:
            for token in self.agent.chat_stream(conversation_id, req.rawQuery):
                yield sse("delta", {"content": token})
        except Exception:
            traceback.print_exc()
            yield sse("error", {"message": "Chat generation failed"})
            return

        if new_conversation:
            try:
                title = self.chat_service.generate_title(
                    self.conversation_memory.get(conversation_id).messages()
                )
                self.conversation_memory.update_conversation_title(conversation_id, title)
                yield sse("conversationTitle", {"conversationId": conversation_id, "title": title})
            except Exception:
                traceback.print_exc()
                yield sse("error", {"message": "Conversation title generation failed"})
                return
        yield sse("done", {})


def main() -> None:
    server = ApiServer()
    server.initialize_agent()
    uvicorn.run(server.app, port=PORT)


if __name__ == "__main__":
    main()
