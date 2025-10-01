# server.py
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from providers.openrouter import OpenRouterProvider
try:
    from providers.deepseek import DeepSeekProvider
    HAVE_DEEPSEEK = True
except Exception:
    HAVE_DEEPSEEK = False

from rag.retrieve import Retriever, format_context
from rag.prompts import build_messages

app = FastAPI(title="UFRO Assistant API", version="1.0.0")

class AskPayload(BaseModel):
    question: str
    provider: str = "openrouter"   # "openrouter" | "deepseek"
    model: Optional[str] = None    # p.ej. "openai/gpt-4.1-mini" o "deepseek-chat"
    k: int = 4
    rag: bool = True
    show_sources: bool = False

def get_llm(provider: str, model: Optional[str]):
    if provider == "openrouter":
        return OpenRouterProvider(model=model or "openai/gpt-4.1-mini")
    if provider == "deepseek":
        if not HAVE_DEEPSEEK:
            raise HTTPException(400, "DeepSeek no está disponible en este despliegue")
        return DeepSeekProvider(model=model or "deepseek-chat")
    raise HTTPException(400, f"Proveedor no soportado: {provider}")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask")
def ask(payload: AskPayload):
    llm = get_llm(payload.provider, payload.model)

    if not payload.rag:
        messages = [
            {"role":"system","content":"Eres un asistente UFRO, responde breve."},
            {"role":"user","content": payload.question},
        ]
        answer = llm.chat(messages)
        return {"answer": answer, "provider": payload.provider, "rag": False}

    # RAG
    retriever = Retriever()
    chunks = retriever.query(payload.question, k=payload.k)
    ctx = format_context(chunks)
    messages = build_messages(payload.question, ctx)
    answer = llm.chat(messages)

    resp: Dict[str, Any] = {"answer": answer, "provider": payload.provider, "rag": True}
    if payload.show_sources:
        srcs = []
        for c in chunks:
            srcs.append({
                "title": c.get("title") or c.get("doc_id"),
                "page": int(c.get("page", 0)) if isinstance(c.get("page", 0),(int,float)) else c.get("page", 0),
                "url": c.get("url",""),
                "snippet": (c.get("text","")[:250]).replace("\n"," ")
            })
        resp["sources"] = srcs
    return resp
