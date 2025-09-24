# rag/retrieve.py
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

INDEX_PATH = Path("data/index.faiss")
META_PATH = Path("data/processed/chunks_meta.parquet")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

@dataclass
class RetrievedChunk:
    score: float
    doc_id: str
    title: str
    page: Any
    url: str
    vigencia: str
    text: str

class Retriever:
    def __init__(self):
        if not INDEX_PATH.exists() or not META_PATH.exists():
            raise FileNotFoundError("Faltan index.faiss o chunks_meta.parquet. Corre rag/embed.py")
        self.index = faiss.read_index(str(INDEX_PATH))
        self.meta = pd.read_parquet(META_PATH)
        self.model = SentenceTransformer(MODEL_NAME)

    def query(self, q: str, k: int = 4) -> List[RetrievedChunk]:
        q_emb = self.model.encode([q], normalize_embeddings=True)
        q_emb = np.asarray(q_emb, dtype="float32")
        scores, idx = self.index.search(q_emb, k)
        results: List[RetrievedChunk] = []
        for score, i in zip(scores[0], idx[0]):
            if i == -1:
                continue
            row = self.meta.iloc[int(i)]
            results.append(RetrievedChunk(
                score=float(score),
                doc_id=str(row["doc_id"]),
                title=str(row["title"]),
                page=row["page"],
                url=str(row["url"]),
                vigencia=str(row["vigencia"]),
                text=str(row["text"]),
            ))
        return results

def format_context(chunks: List[RetrievedChunk]) -> str:
    """
    Devuelve un bloque de contexto con los fragmentos y referencias.
    """
    lines = []
    for c in chunks:
        ref = f"{c.title}, p.{c.page}" if pd.notna(c.page) else f"{c.title}"
        lines.append(f"[Fuente: {ref} | Vigencia: {c.vigencia}]\n{c.text}\n")
    return "\n---\n".join(lines)
