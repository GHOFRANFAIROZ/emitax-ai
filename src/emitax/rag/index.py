"""TF-IDF tabanli RAG indeksi (scikit-learn).

Internet gerektirmez, hafif ve hizlidir. Kucuk korpus icin calisma aninda
kurulabilir; istenirse diske kaydedilip yuklenebilir. (Ileride
sentence-transformers + FAISS'e yukseltmek opsiyoneldir.)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer

from .chunker import Chunk, load_corpus
from .normalize import normalize_text

# Basit Turkce durak (stopword) listesi — cok agresif tutulmadi.
TURKISH_STOPWORDS = [
    "ve", "ile", "bir", "bu", "su", "icin", "olarak", "olan", "da", "de",
    "den", "dan", "ki", "mi", "mu", "cok", "daha", "gibi", "kadar", "gore",
    "ise", "ya", "veya", "ancak", "ama", "fakat", "en", "her", "hem", "ne",
    "olup", "oldugu", "bunlar", "ilgili", "icinde",
]


@dataclass
class RagIndex:
    vectorizer: TfidfVectorizer
    matrix: object            # scipy sparse csr matrisi
    chunks: List[Chunk]

    def __len__(self) -> int:
        return len(self.chunks)


def build_index(chunks: List[Chunk]) -> RagIndex:
    if not chunks:
        raise ValueError("Parca (chunk) listesi bos; indeks kurulamiyor.")
    texts = [c.as_context() for c in chunks]
    vectorizer = TfidfVectorizer(
        preprocessor=normalize_text,
        ngram_range=(1, 2),
        min_df=1,
        stop_words=TURKISH_STOPWORDS,
    )
    matrix = vectorizer.fit_transform(texts)
    return RagIndex(vectorizer=vectorizer, matrix=matrix, chunks=list(chunks))


def build_index_from_dir(docs_dir) -> RagIndex:
    return build_index(load_corpus(docs_dir))


def save_index(index: RagIndex, out_dir) -> None:
    import joblib
    from scipy import sparse

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(index.vectorizer, out_dir / "vectorizer.joblib")
    sparse.save_npz(str(out_dir / "matrix.npz"), index.matrix)
    payload = [c.__dict__ for c in index.chunks]
    (out_dir / "chunks.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_index(in_dir) -> RagIndex:
    import joblib
    from scipy import sparse

    in_dir = Path(in_dir)
    vectorizer = joblib.load(in_dir / "vectorizer.joblib")
    matrix = sparse.load_npz(str(in_dir / "matrix.npz"))
    data = json.loads((in_dir / "chunks.json").read_text(encoding="utf-8"))
    chunks = [Chunk(**d) for d in data]
    return RagIndex(vectorizer=vectorizer, matrix=matrix, chunks=chunks)
