# eval/evaluate.py
import json
import time
import click
from pathlib import Path

# Para resolver imports si se ejecuta con `python -m`
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from providers.openrouter import OpenRouterProvider
try:
    from providers.deepseek import DeepSeekProvider
    HAVE_DEEPSEEK = True
except Exception:
    HAVE_DEEPSEEK = False

from rag.retrieve import Retriever, format_context
from rag.prompts import build_messages


@click.command()
@click.option("--provider", type=click.Choice(["openrouter", "deepseek"]), default="openrouter")
@click.option("--model", default="openai/gpt-4.1-mini")
@click.option("--k", default=4)
@click.option("--gold", default="eval/gold_set.jsonl", help="Ruta al JSONL (una pregunta por línea).")
@click.option("--out", default="eval/results.csv")
@click.option("--limit", default=0, help="Evaluar solo las primeras N líneas (0 = todas).")
@click.option("--verbose/--no-verbose", default=True)
def main(provider, model, k, gold, out, limit, verbose):
    # 0) Proveedor
    if provider == "openrouter":
        llm = OpenRouterProvider(model=model)
    elif provider == "deepseek" and HAVE_DEEPSEEK:
        llm = DeepSeekProvider(model=model)
    else:
        raise click.ClickException(f"Proveedor no soportado o no disponible: {provider}")

    # 1) Retriever
    try:
        retriever = Retriever()
    except Exception as e:
        raise click.ClickException(f"No se pudo inicializar el Retriever. ¿Construiste el índice? "
                                   f"(python rag/ingest.py && python rag/embed.py)\nDetalle: {e}")

    gold_path = Path(gold)
    if not gold_path.exists():
        raise click.ClickException(f"No existe el archivo GOLD: {gold_path}")

    results = []
    total = 0

    # 2) Leer JSONL
    try:
        lines = gold_path.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeDecodeError:
        raise click.ClickException("El archivo GOLD no está en UTF-8. Vuelve a guardarlo con UTF-8.")
    except Exception as e:
        raise click.ClickException(f"No se pudo leer el GOLD: {e}")

    if limit and limit > 0:
        lines = lines[:limit]

    if verbose:
        click.echo(f"[INFO] Evaluando {len(lines)} preguntas desde {gold_path} | k={k} | provider={provider} | model={model}")

    for ln, raw in enumerate(lines, start=1):
        s = raw.strip()
        if not s:
            continue
        try:
            item = json.loads(s)
        except json.JSONDecodeError as e:
            raise click.ClickException(f"JSON inválido en línea {ln}: {e}\nContenido: {s[:200]}")

        # admitir ambos esquemas
        q = item.get("q") or item.get("question")
        expected = item.get("a") or item.get("expected_answer") or ""
        refs = item.get("refs")
        if refs is None:
            exp_doc = item.get("expected_doc")
            refs = [exp_doc] if exp_doc else []

        if not q:
            raise click.ClickException(f"Falta 'q'/'question' en línea {ln}")

        if verbose:
            click.echo(f"\n[Q{ln}] {q}")

        # 3) Recuperación
        try:
            chunks = retriever.query(q, k=k)
            context = format_context(chunks)
        except Exception as e:
            raise click.ClickException(f"Fallo en retrieval (línea {ln}): {e}")

        messages = build_messages(q, context)

        # 4) LLM
        try:
            t0 = time.perf_counter()
            answer = llm.chat(messages)
            latency = time.perf_counter() - t0
        except Exception as e:
            raise click.ClickException(f"Fallo al llamar al LLM (línea {ln}): {e}")

        em = 1 if (expected or "").casefold() in (answer or "").casefold() else 0

        results.append({
            "line": ln,
            "question": q,
            "expected": expected,
            "answer": answer,
            "em": em,
            "latency": round(latency, 2),
            "provider": provider,
            "model": model,
            "expected_refs": "; ".join(refs) if isinstance(refs, list) else str(refs),
        })

        if verbose:
            click.echo(f"[A{ln}] {answer}\n[EM] {em} | [lat] {round(latency,2)}s")

        total += 1

    # 5) Guardar CSV
    if not results:
        click.echo("[AVISO] No se generaron resultados. Revisa tu GOLD y el índice.")
        return

    import csv
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    click.echo(f"\n[OK] {total} filas evaluadas. Resultados en: {out_path}")


if __name__ == "__main__":
    main()
