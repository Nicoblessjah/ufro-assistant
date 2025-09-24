# rag/ingest.py
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import pandas as pd
from pypdf import PdfReader
from pdfminer.high_level import extract_text as pdfminer_extract


RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 3800      # ~equiv. 900-1000 tokens aprox
CHUNK_OVERLAP = 500    # solape para contexto


def read_sources_csv() -> pd.DataFrame:
    src_csv = Path("data/sources.csv")
    if not src_csv.exists():
        raise FileNotFoundError("No se encontró data/sources.csv")
    df = pd.read_csv(src_csv)
    # normaliza columnas esperadas
    required = ["doc_id", "title", "url", "fecha_descarga", "vigencia", "tipo"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Falta la columna '{c}' en data/sources.csv")
    return df


def clean_text(text: str) -> str:
    # Normaliza saltos y espacios
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    # colapsar saltos múltiples
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_pypdf(path: Path) -> List[str]:
    """Devuelve lista de textos por página usando pypdf."""
    pages = []
    try:
        reader = PdfReader(str(path))
        for p in reader.pages:
            t = p.extract_text() or ""
            pages.append(t)
    except Exception as e:
        # Si pypdf falla, devolvemos lista vacía
        return []
    return pages


def extract_pdf_pdfminer(path: Path) -> List[str]:
    """Extrae todo el texto con pdfminer y lo separa por páginas aproximadas."""
    try:
        # pdfminer devuelve texto plano; no segmenta por página fácil.
        # Usamos un separador heurístico si aparece; si no, dejamos una sola "página".
        all_text = pdfminer_extract(str(path)) or ""
    except Exception:
        return []

    # Heurística: pdfminer suele poner form feed \f cuando hay salto de página
    if "\f" in all_text:
        pages = all_text.split("\f")
    else:
        pages = [all_text]
    return pages


def extract_pdf_text(path: Path) -> Tuple[List[str], bool]:
    """
    Intenta pypdf; si la mayoría de páginas sale vacía, intenta pdfminer.
    Devuelve (pages, needs_ocr) donde pages es lista de textos por página y
    needs_ocr=True si no se pudo extraer texto razonable (posible PDF escaneado).
    """
    pages = extract_pdf_pypdf(path)
    # cuenta páginas con texto "suficiente"
    non_empty = sum(1 for t in pages if t and len(t.strip()) > 30)
    if pages and non_empty >= max(1, int(0.3 * len(pages))):
        return pages, False  # pypdf funcionó razonablemente

    # Fallback con pdfminer
    pm_pages = extract_pdf_pdfminer(path)
    non_empty_pm = sum(1 for t in pm_pages if t and len(t.strip()) > 30)
    if pm_pages and non_empty_pm >= 1:
        return pm_pages, False

    # Nada legible → probablemente escaneado (OCR requerido)
    return [], True


def extract_txt(path: Path) -> List[str]:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    txt = clean_text(txt)
    # lo tratamos como "una sola página"
    return [txt]


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Chunking por caracteres. Devuelve lista de ventanas con solape.
    """
    text = text.strip()
    n = len(text)
    if n <= size:
        return [text]

    chunks = []
    start = 0
    while start < n:
        end = min(n, start + size)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def build_chunks_for_doc(
    row: pd.Series, path: Path
) -> List[Dict]:
    """
    Dado un registro de sources.csv y el archivo físico,
    devuelve lista de dicts con metadatos + chunk.text.
    """
    ext = path.suffix.lower()
    needs_ocr = False
    page_texts: List[str] = []

    if ext == ".pdf":
        page_texts, needs_ocr = extract_pdf_text(path)
    elif ext in [".txt", ".text"]:
        page_texts = extract_txt(path)
    elif ext in [".html", ".htm"]:
        # lectura simple de HTML como texto plano (rápido);
        # más adelante se puede usar bs4 para limpiar etiquetas.
        html = path.read_text(encoding="utf-8", errors="ignore")
        page_texts = [clean_text(re.sub(r"<[^>]+>", " ", html))]
    else:
        print(f"[WARN] Extensión no soportada: {path.name}")
        return []

    chunks_rows: List[Dict] = []
    if needs_ocr:
        # Genera un "placeholder" para registrar que el doc requiere OCR.
        chunks_rows.append({
            "doc_id": row["doc_id"],
            "title": row["title"],
            "page": None,
            "url": row.get("url", ""),
            "vigencia": row.get("vigencia", ""),
            "text": "",
            "source_path": str(path),
            "needs_ocr": True,
        })
        return chunks_rows

    for i, page_text in enumerate(page_texts, start=1):
        page_text = clean_text(page_text or "")
        if not page_text or len(page_text) < 30:
            continue
        for ch in chunk_text(page_text):
            chunks_rows.append({
                "doc_id": row["doc_id"],
                "title": row["title"],
                "page": i,
                "url": row.get("url", ""),
                "vigencia": row.get("vigencia", ""),
                "text": ch,
                "source_path": str(path),
                "needs_ocr": False,
            })
    return chunks_rows


def main():
    df_src = read_sources_csv()

    # Mapa doc_id -> ruta física esperada (por nombre razonable)
    # Buscamos por coincidencia parcial del doc_id en el nombre del archivo.
    files = list(RAW_DIR.glob("*"))
    if not files:
        print("No hay archivos en data/raw/")
        sys.exit(1)

    all_rows: List[Dict] = []
    missing: List[str] = []

    for _, row in df_src.iterrows():
        doc_id = str(row["doc_id"])
        # Busca un archivo cuyo nombre contenga el doc_id o alguna palabra clave del title
        candidate: Optional[Path] = None

        # 1) intento por doc_id
        for f in files:
            if doc_id.lower() in f.stem.lower():
                candidate = f
                break

        # 2) intento heurístico por título
        if candidate is None:
            title_key = re.sub(r"[^a-z0-9]+", " ", str(row["title"]).lower())
            title_key = "_".join(title_key.split()[:3])  # primeras palabras
            for f in files:
                if title_key and title_key in f.stem.lower():
                    candidate = f
                    break

        if candidate is None:
            print(f"[WARN] No se encontró archivo para doc_id={doc_id}. Renombra el archivo en data/raw/ para que contenga el doc_id.")
            missing.append(doc_id)
            continue

        print(f"[INFO] Procesando {doc_id} -> {candidate.name}")
        rows = build_chunks_for_doc(row, candidate)
        all_rows.extend(rows)

    if not all_rows:
        print("No se generaron chunks. Revisa si los PDFs requieren OCR.")
        sys.exit(1)

    df_chunks = pd.DataFrame(all_rows)
    out_path = OUT_DIR / "chunks.parquet"
    df_chunks.to_parquet(out_path, index=False)
    print(f"[OK] Guardado {out_path} con {len(df_chunks)} chunks.")

    needs = df_chunks[df_chunks["needs_ocr"] == True]
    if not needs.empty:
        print("\n[AVISO] Algunos documentos parecen requerir OCR (no se pudo extraer texto):")
        print(needs[["doc_id", "title", "source_path"]].drop_duplicates().to_string(index=False))
        print("Más adelante podemos activar OCR con pytesseract si hace falta.")


if __name__ == "__main__":
    main()
