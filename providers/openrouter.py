import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
from .base import Provider


class OpenRouterProvider(Provider):
    name = "openrouter"

    def __init__(self, model: str = "openai/gpt-4.1-mini"):
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("Falta OPENROUTER_API_KEY en tu .env")

        # Cliente OpenAI apuntando a OpenRouter (OpenAI-compatible)
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        self.model = model

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", 0.2),
            max_tokens=kwargs.get("max_tokens", 128),
        )
        return (resp.choices[0].message.content or "").strip()
