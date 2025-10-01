from dotenv import load_dotenv; load_dotenv()
import os
from openai import OpenAI

print("Key prefix:", (os.getenv("OPENROUTER_API_KEY") or "")[:12])

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    default_headers={
        "HTTP-Referer": os.getenv("OPENROUTER_REFERER","http://localhost"),
        "X-Title": os.getenv("OPENROUTER_TITLE","UFRO Assistant"),
    },
)

r = client.chat.completions.create(
    model="openai/gpt-4.1-mini",
    messages=[{"role":"user","content":"Di ok"}],
    max_tokens=5
)
print("Respuesta:", r.choices[0].message.content)
