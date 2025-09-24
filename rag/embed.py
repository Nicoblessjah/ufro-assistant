# rag/embed.py
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path("data/processed/chunks.parquet")
INDEX_PATH = Path("data/index.faiss")
META_PATH = Path("data/processed/chunks_meta.parquet")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim

def main():
    t0 = time.time()
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError("No existe data/processed/chunks.parquet. Ejecuta primero rag/ingest.py")

    df = pd.read_parquet(CHUNKS_PATH)
    # Usar solo los chunks que tienen texto (excluir los marcados needs_ocr=True)
    df = df[(~df["needs_ocr"]) & (df["text"].str.len() > 0)].copy()
    if df.empty:
        raise RuntimeError("No hay chunks con texto utilizable. ¿Todos requieren OCR?")

    print(f"[INFO] Embeddings para {len(df)} chunks...")
    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode(df["text"].tolist(), normalize_embeddings=True, show_progress_bar=True)
    emb = np.asarray(emb, dtype="float32")

    # FAISS: producto interno (coseno si normalizamos)
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    # Persistir
    faiss.write_index(index, str(INDEX_PATH))
    # Guardar metadatos en paralelo (mismo orden que emb)
    meta_cols = ["doc_id", "title", "page", "url", "vigencia", "source_path", "text"]
    df[meta_cols].to_parquet(META_PATH, index=False)

    dt = time.time() - t0
    print(f"[OK] Guardado índice {INDEX_PATH} y metadatos {META_PATH} en {dt:.2f}s.")

if __name__ == "__main__":
    main()
