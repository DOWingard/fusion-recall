"""fusion-recall: a drop-in hybrid-retrieval layer.

Wraps any existing dense/embedding retriever and lifts it to full fusion recall —
parallel BM25 lexical recall, rank fusion, and cross-encoder reranking — returning
ranked results plus a per-source-provenance score vector suitable for a downstream
confidence/entropy gate.

This module re-exports the entire stable public surface;
everything not listed in ``__all__`` is internal.
"""

from __future__ import annotations

from fusion_recall.bm25 import BM25Index, default_tokenizer
from fusion_recall.contracts import (
    DEFAULT_CE_MODEL,
    DEFAULT_PROFILE,
    SOURCE_DENSE,
    SOURCE_LEXICAL,
    CrossEncoder,
    DenseRetriever,
    EmptyCorpusError,
    Fuser,
    FusionError,
    FusionProfile,
    HybridResult,
    InvalidProfileError,
    LexicalRetriever,
    Normalize,
    TextLookup,
    Tokenizer,
)
from fusion_recall.fusion import rrf, weighted_fuse
from fusion_recall.hybrid import HybridRetriever, lift
from fusion_recall.rerank import NoOpCrossEncoder, SentenceTransformerCrossEncoder

__version__ = "0.0.0"

__all__ = [
    # Entry points
    "lift",
    "HybridRetriever",
    # Built-ins
    "BM25Index",
    "default_tokenizer",
    "rrf",
    "weighted_fuse",
    "SentenceTransformerCrossEncoder",
    "NoOpCrossEncoder",
    # Value objects
    "FusionProfile",
    "HybridResult",
    # Constants
    "DEFAULT_PROFILE",
    "DEFAULT_CE_MODEL",
    "SOURCE_DENSE",
    "SOURCE_LEXICAL",
    # Protocols (seams)
    "DenseRetriever",
    "LexicalRetriever",
    "CrossEncoder",
    "Fuser",
    "TextLookup",
    "Tokenizer",
    "Normalize",
    # Exceptions
    "FusionError",
    "EmptyCorpusError",
    "InvalidProfileError",
]
