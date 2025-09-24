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
def main(question: str, provider: str, model: str, k: int, rag: bool):
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
        # Modo directo (sin recuperación)
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": "Eres un asistente UFRO, responde breve."},
            {"role": "user", "content": question},
        ]
        answer = llm.chat(messages)
        click.secho("\nRespuesta (sin RAG):", fg="yellow")
        click.echo(answer)
        return

    # RAG: recuperar contexto y pasar al LLM con política de citas/abstención
    try:
        retriever = Retriever()
    except Exception as e:
        raise click.ClickException(f"No se pudo inicializar el retriever: {e}")

    chunks = retriever.query(question, k=k)
    context_block = format_context(chunks)
    messages = build_messages(question, context_block)

    answer = llm.chat(messages)
    click.secho("\nRespuesta (RAG):", fg="green")
    click.echo(answer)

if __name__ == "__main__":
    main()
