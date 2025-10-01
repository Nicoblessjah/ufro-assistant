from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from fastapi import Form

from providers.openrouter import OpenRouterProvider
try:
    from providers.deepseek import DeepSeekProvider
    HAVE_DEEPSEEK = True
except Exception:
    HAVE_DEEPSEEK = False

from rag.retrieve import Retriever, format_context
from rag.prompts import build_messages

app = FastAPI(title="UFRO Assistant API", version="1.0.0")

# Sirve archivos estáticos (como CSS, JS) en el directorio "static"
app.mount("/static", StaticFiles(directory="static"), name="static")

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

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/ask")
async def ask(question: str = Form(...), provider: str = Form(...), model: Optional[str] = Form(None), 
              k: int = Form(4), rag: bool = Form(True), show_sources: bool = Form(False)):
    
    llm = get_llm(provider, model)

    if not rag:
        messages = [
            {"role":"system","content":"Eres un asistente UFRO, responde breve."},
            {"role":"user","content": question},
        ]
        answer = llm.chat(messages)
        return {"answer": answer, "provider": provider, "rag": False}

    retriever = Retriever()
    chunks = retriever.query(question, k=k)
    context_block = format_context(chunks)
    messages = build_messages(question, context_block)

    answer = llm.chat(messages)

    resp: Dict[str, Any] = {"answer": answer, "provider": provider, "rag": True}
    
    if show_sources:
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

@app.get("/health")
def health():
    return {"status": "ok"}
