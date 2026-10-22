import os

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

TITLE_SYSTEM_PROMPT = (
    "你是一个会话标题生成器。根据对话内容生成一个简短的标题（不超过20个字），"
    "只输出标题本身，不要加引号、标点或任何解释。"
)
TITLE_REQUEST = "请为以上对话生成一个简短的标题。"


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
