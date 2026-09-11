"""Vektor veritabani (embeddings + FAISS) -- anlamsal arama.

TF-IDF kelime eslesmesi yapar; bu modul ANLAM eslesmesi yapar: her parca cok
dilli bir model ile vektore cevrilir, FAISS ile kosinus benzerligine gore aranir.
Model yereldir (sentence-transformers); ilk indirme internet ister, sonra
cevrimdisi calisir.

Kurulum (bir kez):
    pip install sentence-transformers faiss-cpu
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from .chunker import Chunk, load_corpus

# Cok dilli, Turkce destekli, hafif ve hizli model (384 boyut, ~470 MB).
DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


@dataclass
class EmbedIndex:
    chunks: List[Chunk]
    faiss_index: object                 # faiss.IndexFlatIP
    model_name: str
    dim: int
    model: object = field(default=None, repr=False)  # SentenceTransformer (lazy)

    def __len__(self) -> int:
        return len(self.chunks)

    def _ensure_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
        return self.model

    def encode(self, texts: List[str]) -> np.ndarray:
        model = self._ensure_model()
        vecs = model.encode(texts, normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(vecs, dtype="float32")


def _build_faiss(vectors: np.ndarray):
    import faiss
    vectors = np.ascontiguousarray(vectors.astype("float32"))
    faiss.normalize_L2(vectors)          # guvenlik icin tekrar normalize
    idx = faiss.IndexFlatIP(vectors.shape[1])   # ic carpim = normalize vektorde kosinus
    idx.add(vectors)
    return idx


def build_index_from_vectors(chunks: List[Chunk], vectors: np.ndarray,
                             model_name: str = DEFAULT_MODEL) -> EmbedIndex:
    if len(chunks) != vectors.shape[0]:
        raise ValueError("parca sayisi ile vektor sayisi uyusmuyor")
    faiss_index = _build_faiss(vectors)
    return EmbedIndex(chunks=list(chunks), faiss_index=faiss_index,
                      model_name=model_name, dim=int(vectors.shape[1]))


def build_embed_index(chunks: List[Chunk], model_name: str = DEFAULT_MODEL,
                      batch_size: int = 64) -> EmbedIndex:
    if not chunks:
        raise ValueError("parca listesi bos")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    texts = [c.as_context() for c in chunks]
    vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                        batch_size=batch_size, show_progress_bar=True)
    vecs = np.asarray(vecs, dtype="float32")
    idx = build_index_from_vectors(chunks, vecs, model_name)
    idx.model = model                    # sorgu icin modeli sakla
    return idx


def build_embed_index_from_dir(corpus_dir, model_name: str = DEFAULT_MODEL) -> EmbedIndex:
    return build_embed_index(load_corpus(corpus_dir), model_name=model_name)


def save_embed_index(index: EmbedIndex, out_dir) -> None:
    import faiss
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index.faiss_index, str(out_dir / "vectors.faiss"))
    (out_dir / "chunks.json").write_text(
        json.dumps([c.__dict__ for c in index.chunks], ensure_ascii=False),
        encoding="utf-8")
    (out_dir / "meta.json").write_text(
        json.dumps({"model_name": index.model_name, "dim": index.dim}, ensure_ascii=False),
        encoding="utf-8")


def load_embed_index(in_dir) -> EmbedIndex:
    import faiss
    in_dir = Path(in_dir)
    faiss_index = faiss.read_index(str(in_dir / "vectors.faiss"))
    data = json.loads((in_dir / "chunks.json").read_text(encoding="utf-8"))
    meta = json.loads((in_dir / "meta.json").read_text(encoding="utf-8"))
    chunks = [Chunk(**d) for d in data]
    return EmbedIndex(chunks=chunks, faiss_index=faiss_index,
                      model_name=meta["model_name"], dim=meta["dim"])


def get_embed_index(corpus_dir, cache_dir=None, model_name: str = DEFAULT_MODEL) -> "EmbedIndex":
    """Kayitli vektor indeksi varsa yukler; yoksa kurar ve kaydeder.
    Ilk kurulum modeli indirir (~470 MB, internet) + vektorleri hesaplar (birkac dk).
    Sonraki calismalar cache'ten hizli yuklenir. Korpus degisirse cache_dir silinmeli.
    """
    from pathlib import Path
    cache_dir = Path(cache_dir) if cache_dir else Path(corpus_dir) / ".embed_index"
    if (cache_dir / "vectors.faiss").exists() and (cache_dir / "chunks.json").exists():
        return load_embed_index(cache_dir)
    index = build_embed_index_from_dir(corpus_dir, model_name=model_name)
    try:
        save_embed_index(index, cache_dir)
    except Exception:
        pass
    return index
