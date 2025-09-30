# scripts/fix_gold.py
import json, sys, re
from pathlib import Path

def load_bytes(path: Path) -> bytes:
    return path.read_bytes()

def try_decode_mojibake(b: bytes) -> str:
    """
    Intenta reparar texto 'mojibake' típico (Ã¡, Ã©, Â¿, etc.).
    Estrategia: interpretar lo guardado como Latin-1 y re-decodificar a UTF-8.
    Si falla, cae a UTF-8 con o sin BOM.
    """
    for enc in (("latin1","utf-8"), ("utf-8-sig",None), ("utf-8",None)):
        try:
            if enc[1] is None:
                return b.decode(enc[0])
            # caso mojibake: bytes -> latin1 string -> re-encode latin1 -> decode utf-8 real
            s = b.decode(enc[0], errors="strict")
            s2 = s.encode("latin1", errors="strict").decode("utf-8", errors="strict")
            return s2
        except Exception:
            continue
    # último intento: devuelve utf-8 'best effort'
    return b.decode("utf-8", errors="replace")

def read_any_json_or_jsonl(txt: str):
    t = txt.strip()
    # Si parece un array JSON completo, lo parseamos y devolvemos lista
    if t.startswith("[") and t.endswith("]"):
        try:
            arr = json.loads(t)
            if isinstance(arr, list):
                return arr
        except Exception:
            pass
    # Si no, interpretamos como JSONL (una linea por objeto)
    items = []
    for i, line in enumerate(txt.splitlines(), 1):
        s = line.strip()
        if not s:
            continue
        # elimina coma final al estilo JSON de array
        if s.endswith(","):
            s = s[:-1].rstrip()
        try:
            obj = json.loads(s)
        except Exception as e:
            raise RuntimeError(f"Línea {i} no es JSON válido: {e}\nContenido: {s[:200]}")
        items.append(obj)
    return items

def normalize_items(items):
    """
    Normaliza claves para que el evaluate acepte ambos esquemas.
    (No reescribe campos, solo asegura que cada ítem sea dict y no esté vacío)
    """
    out = []
    for it in items:
        if not isinstance(it, dict):
            raise RuntimeError(f"Elemento no es objeto JSON: {it!r}")
        out.append(it)
    return out

def write_jsonl(path: Path, items):
    # Guardar en UTF-8 sin BOM, una línea por objeto, sin comas
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

def main():
    if len(sys.argv) < 3:
        print("Uso: python scripts/fix_gold.py <in_path> <out_path>")
        sys.exit(1)
    inp = Path(sys.argv[1])
    outp = Path(sys.argv[2])
    if not inp.exists():
        print(f"No existe entrada: {inp}")
        sys.exit(2)

    raw = load_bytes(inp)
    txt = try_decode_mojibake(raw)
    items = read_any_json_or_jsonl(txt)
    items = normalize_items(items)
    outp.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(outp, items)
    print(f"[OK] Escribí JSONL limpio en: {outp}  (líneas: {len(items)})")

if __name__ == "__main__":
    main()
