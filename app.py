from dotenv import load_dotenv
load_dotenv()

import click
from typing import List, Dict

from providers.openrouter import OpenRouterProvider
from rag.retrieve import Retriever, format_context
from rag.prompts import build_messages

@click.command()
@click.argument("question", required=False)
@click.option("--provider", type=click.Choice(["openrouter"]), default="openrouter",
              help="Proveedor LLM a usar.")
@click.option("--model", default="openai/gpt-4.1-mini", help="Modelo del proveedor.")
@click.option("--k", default=4, show_default=True, help="Top-k de fragmentos a recuperar.")
@click.option("--rag/--no-rag", default=True, show_default=True, help="Usar RAG (recuperación + citas).")
@click.option("--show-sources/--no-show-sources", default=False, show_default=True,   # <--- NUEVO
              help="Muestra las fuentes (chunks) recuperadas.")
def main(question: str, provider: str, model: str, k: int, rag: bool, show_sources: bool):   # <--- agrega show_sources
    """
    CLI para hacer preguntas. Ejemplos:
      python app.py "¿Cuál es la fecha de inicio del semestre 2025?"
      python app.py "¿Cómo apelar una nota?" --k 5
      python app.py --no-rag "Hola, responde OK"
    """
    if provider == "openrouter":
        llm = OpenRouterProvider(model=model)
    else:
        raise click.ClickException(f"Proveedor no soportado: {provider}")

    if not question:
        question = click.prompt("Escribe tu pregunta")

    if not rag:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": "Eres un asistente UFRO, responde breve."},
            {"role": "user", "content": question},
        ]
        answer = llm.chat(messages)
        click.secho("\nRespuesta (sin RAG):", fg="yellow")
        click.echo(answer)
        return

    # RAG
    try:
        retriever = Retriever()
    except Exception as e:
        raise click.ClickException(f"No se pudo inicializar el retriever: {e}")

    chunks = retriever.query(question, k=k)

    # Mostrar fuentes recuperadas para depurar
    if show_sources:  # <-- Asegúrate de que este bloque esté dentro de la función main()
        click.secho("\nFuentes recuperadas:", fg="cyan")
    for i, d in enumerate(chunks, 1):
        # Reemplaza d.get() por acceso directo a las propiedades del fragmento
        page = getattr(d, "page", 0)
        title = getattr(d, "title", getattr(d, "doc_id", ""))
        url = getattr(d, "url", "")
        snippet = (getattr(d, "text", "").replace("\n", " ")[:250] + ("..." if len(getattr(d, "text", "")) > 250 else ""))
        click.echo(f"[{i}] {title} (p.{page}) -> {url}")
        click.echo(f"     {snippet}\n")

    context_block = format_context(chunks)
    messages = build_messages(question, context_block)

    answer = llm.chat(messages)
    click.secho("\nRespuesta (RAG):", fg="green")
    click.echo(answer)

if __name__ == "__main__":
    main()
