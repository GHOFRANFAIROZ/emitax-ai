"""Emitax RAG katmani -- gercek resmi mevzuat uzerinde anlamsal (vektor) arama."""
from .chunker import Chunk, load_corpus, GROUP_SOURCE
from .embed_index import (EmbedIndex, build_embed_index, build_embed_index_from_dir,
                          build_index_from_vectors, save_embed_index, load_embed_index,
                          get_embed_index, DEFAULT_MODEL)
from .retrieve import RetrievedChunk, retrieve
from .recommend import Recommendation, recommend, recommend_all, build_query

__all__ = [
    "Chunk", "load_corpus", "GROUP_SOURCE",
    "EmbedIndex", "build_embed_index", "build_embed_index_from_dir",
    "build_index_from_vectors", "save_embed_index", "load_embed_index",
    "get_embed_index", "DEFAULT_MODEL",
    "RetrievedChunk", "retrieve",
    "Recommendation", "recommend", "recommend_all", "build_query",
]
