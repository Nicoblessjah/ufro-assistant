# rag/ingest.py
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import pandas as pd
from pypdf import PdfReader

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 3800      # ~equiv. 900-1000 tokens aprox
CHUNK_OVERLAP = 500    # solape para contexto


# ==============================
# Lectura de documentos (según tutorial del profesor)
# ==============================
def read_txt(path: Path) -> str:
    return open(path, "r", encoding="utf-8", errors="ignore").read()

def read_pdf(path: Path) -> str:
    try:
        return "\n".join([p.extract_text() or "" for p in PdfReader(str(path)).pages])
    except Exception:
        return ""

def load_docs(data_dir: Path) -> List[Tuple[Path, str]]:
    """
    Recorre data/raw y devuelve [(path, texto)]. A diferencia del original,
    ahora SIEMPRE agregamos el archivo, aunque el texto esté vacío,
    para poder marcar needs_ocr más adelante y no perder trazabilidad.
    """
    docs: List[Tuple[Path, str]] = []
    for root, _, files in os.walk(data_dir):
        for fn in files:
            path = Path(root) / fn
            if fn.lower().endswith(".txt"):
                txt = read_txt(path)
            elif fn.lower().endswith(".pdf"):
                txt = read_pdf(path)
            else:
                continue
            txt = re.sub(r"\n{3,}", "\n\n", txt).strip()
            # SIEMPRE anexamos; si viene vacío, luego se marca needs_ocr=True
            docs.append((path, txt))
    return docs


# ==============================
# Utilidades adicionales
# ==============================
def read_sources_csv() -> pd.DataFrame:
    src_csv = Path("data/sources.csv")
    if not src_csv.exists():
        raise FileNotFoundError("No se encontró data/sources.csv")
    df = pd.read_csv(src_csv)
    required = ["doc_id", "title", "url", "fecha_descarga", "vigencia", "tipo"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Falta la columna '{c}' en data/sources.csv")
    return df

def clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = text.strip()
    n = len(text)
    if n <= size:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < n:
        end = min(n, start + size)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


# ==============================
# Construcción de chunks con metadatos
# ==============================
def build_chunks_for_doc(row: pd.Series, path: Path, full_text: str) -> List[Dict]:
    """
    Construye los registros de chunks para un documento. Si no hay texto utilizable,
    devuelve un placeholder con needs_ocr=True para mantener trazabilidad.
    """
    chunks_rows: List[Dict] = []
    full_text = clean_text(full_text or "")

    # Si no hay texto -> marcar como OCR requerido
    if not full_text or len(full_text) < 30:
        return [{
            "doc_id": row["doc_id"],
            "title": row["title"],
            "page": None,
            "url": row.get("url", ""),
            "vigencia": row.get("vigencia", ""),
            "text": "",
            "source_path": str(path),
            "needs_ocr": True,
        }]

    # En este esquema no separamos por páginas, sino por documento completo
    page = 1
    for ch in chunk_text(full_text):
        chunks_rows.append({
            "doc_id": row["doc_id"],
            "title": row["title"],
            "page": page,
            "url": row.get("url", ""),
            "vigencia": row.get("vigencia", ""),
            "text": ch,
            "source_path": str(path),
            "needs_ocr": False,
        })
    return chunks_rows


# ==============================
# MAIN
# ==============================
def main():
    df_src = read_sources_csv()
    docs = load_docs(RAW_DIR)
    if not docs:
        print("No hay archivos válidos en data/raw/")
        sys.exit(1)

    all_rows: List[Dict] = []
    missing: List[str] = []

    for _, row in df_src.iterrows():
        doc_id = str(row["doc_id"]).lower()
        candidate: Optional[Tuple[Path, str]] = None

        # Busca coincidencia por doc_id en el nombre del archivo
        for (path, txt) in docs:
            if doc_id in path.stem.lower():
                candidate = (path, txt)
                break

        if candidate is None:
            print(f"[WARN] No se encontró archivo para doc_id={doc_id}. Renombra el archivo en data/raw/ para que contenga el doc_id.")
            missing.append(doc_id)
            continue

        print(f"[INFO] Procesando {row['doc_id']} -> {candidate[0].name}")
        rows = build_chunks_for_doc(row, candidate[0], candidate[1])
        all_rows.extend(rows)

    if not all_rows:
        print("No se generaron chunks. Revisa si los documentos contienen texto legible.")
        sys.exit(1)

    df_chunks = pd.DataFrame(all_rows)
    out_path = OUT_DIR / "chunks.parquet"
    df_chunks.to_parquet(out_path, index=False)
    print(f"[OK] Guardado {out_path} con {len(df_chunks)} chunks.")

if __name__ == "__main__":
    main()
