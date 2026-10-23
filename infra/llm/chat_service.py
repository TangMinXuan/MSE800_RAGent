import os

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

TITLE_SYSTEM_PROMPT = """Generate a concise title for this conversation based on its history.
Use the same language as the conversation. Return only the title, without quotes or explanation.
"""
TITLE_REQUEST = "Generate a concise title for the conversation above."


class ChatService:
    def __init__(self, model: ChatOpenAI | None = None):
        self.model = model or ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("API_KEY"),
            model="openai/gpt-4o-mini",
        )

    def chat(self, prompt: "str | list[BaseMessage]") -> str:
        if isinstance(prompt, str):
            prompt = [HumanMessage(content=prompt)]
        return self.model.invoke(prompt).content

    def generate_title(self, history: list[BaseMessage]) -> str:
        messages = [SystemMessage(content=TITLE_SYSTEM_PROMPT), *history, HumanMessage(content=TITLE_REQUEST)]
        return self.chat(messages).strip()
