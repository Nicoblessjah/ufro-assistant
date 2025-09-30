# providers/deepseek.py
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
from .base import Provider

load_dotenv()

class DeepSeekProvider(Provider):
    name = "deepseek"

    def __init__(self, model: str = "deepseek-chat"):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("Falta DEEPSEEK_API_KEY en tu .env")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )
        self.model = model

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", 0.2),
            max_tokens=kwargs.get("max_tokens", 256),
        )
        return (resp.choices[0].message.content or "").strip()
