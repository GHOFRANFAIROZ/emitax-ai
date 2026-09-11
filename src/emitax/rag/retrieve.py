"""Sorguya en yakin yasal parcalari getirir.

Iki indeks tipini de destekler:
- EmbedIndex (vektor DB / FAISS)  -> anlamsal arama (varsayilan, onerilen)
- RagIndex   (TF-IDF)             -> kelime eslesmesi (yedek)
Sektor filtresi, grup boost ve only_groups her iki tipte de ayni uygulanir.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from .chunker import Chunk


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


def _allowed(chunk: Chunk, sectors: Optional[Sequence[str]]) -> bool:
    if chunk.sector_independent:
        return True
    if not sectors:
        return False
    return any(s in chunk.sectors for s in sectors)


def _postfilter(pairs, sectors, boost_groups, only_groups, k):
    """(chunk, skor) listesine sektor/only_groups filtresi + grup boost uygular."""
    ranked = []
    for c, score in pairs:
        if not _allowed(c, sectors):
            continue
        if only_groups and c.group not in only_groups:
            continue
        mult = boost_groups.get(c.group, 1.0) if boost_groups else 1.0
        rank_score = score * mult          # SADECE siralama icin
        ranked.append((c, score, rank_score))
    ranked.sort(key=lambda x: -x[2])       # boost'lu skora gore sirala
    # gorunen skor gercek benzerliktir (0..1), boost carpani gizli kalir
    return [RetrievedChunk(chunk=c, score=float(min(max(score, 0.0), 1.0)))
            for c, score, _ in ranked[:k]]


def _retrieve_embed(index, query, k, sectors, boost_groups, only_groups):
    import numpy as np
    qv = index.encode([query])                      # (1, d) normalize
    qv = np.ascontiguousarray(qv.astype("float32"))
    n = min(max(k * 60, 400), len(index.chunks))     # filtre + boost icin genis cek
    scores, idxs = index.faiss_index.search(qv, n)
    pairs = [(index.chunks[int(i)], float(sc))
             for sc, i in zip(scores[0], idxs[0]) if i >= 0 and sc > 0]
    return _postfilter(pairs, sectors, boost_groups, only_groups, k)


def _retrieve_tfidf(index, query, k, sectors, boost_groups, only_groups):
    from sklearn.metrics.pairwise import linear_kernel
    q_vec = index.vectorizer.transform([query])
    sims = linear_kernel(q_vec, index.matrix).ravel()
    order = sims.argsort()[::-1]
    pairs = []
    for i in order:
        idx = int(i)
        if sims[idx] <= 0:
            break
        pairs.append((index.chunks[idx], float(sims[idx])))
        if len(pairs) >= max(k * 60, 400):
            break
    return _postfilter(pairs, sectors, boost_groups, only_groups, k)


def retrieve(index, query: str, k: int = 3,
             sectors: Optional[Sequence[str]] = None,
             boost_groups: Optional[dict] = None,
             only_groups: Optional[Sequence[str]] = None) -> List[RetrievedChunk]:
    if not query or not str(query).strip():
        return []
    if hasattr(index, "faiss_index"):
        return _retrieve_embed(index, query, k, sectors, boost_groups, only_groups)
    return _retrieve_tfidf(index, query, k, sectors, boost_groups, only_groups)
