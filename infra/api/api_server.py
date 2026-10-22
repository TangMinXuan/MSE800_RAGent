import json
from typing import Any, Iterator

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

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


def _is_blank(s: str | None) -> bool:
    return not (s and s.strip())


def sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class ApiServer:
    def __init__(self, agent: Any = None, conversation_memory: Any = None, chat_service: Any = None):
        self.agent = agent
        self.conversation_memory = conversation_memory
        self.chat_service = chat_service
        self.app = self._build_app()

    def _build_app(self) -> FastAPI:
        app = FastAPI()

        @app.get("/health")
        def health():
            return {"status": "ok"}

        @app.post("/api/chat", response_model=ChatResp)
        def chat(req: ChatReq):
            return ChatResp(
                code=200,
                msg="success",
                conversationId=req.conversationId or "test-conversation-id",
                conversationTitle="test title",
                llmResp="test response",
            )

        @app.post("/api/chat/stream")
        def chat_stream(req: ChatReq):
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
        if _is_blank(req.rawQuery):
            yield sse("error", {"msg": "rawQuery must not be blank"})
            yield sse("done", {})
            return
        if self.agent is None:
            yield sse("error", {"msg": "Agent not wired yet"})
            yield sse("done", {})
            return
        # Real agent wiring is deferred until pipeline/Agent are migrated.
        yield sse("error", {"msg": "Agent not wired yet"})
        yield sse("done", {})


def main() -> None:
    uvicorn.run(ApiServer().app, port=PORT)


if __name__ == "__main__":
    main()
