# rag/prompts.py
from typing import List, Dict

def system_prompt() -> str:
    return (
        "Eres 'Asistente UFRO', experto en normativa vigente. "
        "Usa SOLO la información de los fragmentos proporcionados. "
        "Si no hay evidencia suficiente, responde literalmente: "
        "'No encontrado en normativa UFRO' y sugiere la oficina/unidad correspondiente. "
        "Al final incluye SIEMPRE una sección 'Referencias' con el formato "
        "[Documento, p.xx] o [Documento, sección]. Sé conciso y transparente sobre la vigencia."
    )

def build_messages(question: str, context_block: str) -> List[Dict[str, str]]:
    user = (
        f"Pregunta:\n{question}\n\n"
        "Contexto (fragmentos recuperados con metadatos):\n"
        f"{context_block}\n\n"
        "Instrucciones de respuesta:\n"
        "- Responde en 1–2 párrafos claros.\n"
        "- No inventes. Si no hay evidencia, abstente como se indicó.\n"
        "- Al final agrega 'Referencias' con [Documento, p.xx]."
    )
    return [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": user},
    ]
