"""Core types, seams, exceptions, and constants for fusion-recall.

This module is foundational: every other module imports the ``ID``/``Ranking``
aliases, the seven runtime-checkable Protocols, the exception hierarchy,
``FusionProfile``/``HybridResult``, and the ``SOURCE_*`` / ``DEFAULT_*``
constants from here. It contains no logic beyond these definitions so the
contract surface stays stable and parallel work can proceed against it.

There is no ``ce_blend``: the cross-encoder replaces
scores within the rerank window rather than blending them.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Protocol, runtime_checkable

import numpy as np

# --------------------------------------------------------------------------- #
# Canonical source keys. Defined once and referenced everywhere (the weights
# default, the retrieve rankings dict, and the provenance columns) so there are
# no repeated magic strings.
# --------------------------------------------------------------------------- #
SOURCE_DENSE = "dense"
SOURCE_LEXICAL = "lexical"

# --------------------------------------------------------------------------- #
# Type aliases.
# --------------------------------------------------------------------------- #
ID = Hashable
# An ordered (descending by score) tuple of (id, score) pairs.
Ranking = tuple[tuple[ID, float], ...]


# --------------------------------------------------------------------------- #
# Seams. Every swappable component is a runtime-checkable Protocol so a host can
# inject its own implementation and the library can structurally validate it.
# --------------------------------------------------------------------------- #
@runtime_checkable
class DenseRetriever(Protocol):
    """Maps a text query to ranked ``(id, score)`` results (opaque to vectors)."""

    def __call__(self, query: str, k: int) -> Ranking: ...


@runtime_checkable
class LexicalRetriever(Protocol):
    """Maps a text query to ranked ``(id, score)`` results via lexical recall."""

    def __call__(self, query: str, k: int) -> Ranking: ...


@runtime_checkable
class CrossEncoder(Protocol):
    """Re-scores ``(query, doc)`` pairs; returns an array of shape ``(len(docs),)``."""

    def score(self, query: str, docs: Sequence[str]) -> np.ndarray: ...


@runtime_checkable
class Fuser(Protocol):
    """Merges per-source rankings (keyed by source name) into one ranking."""

    def __call__(self, rankings: Mapping[str, Ranking]) -> Ranking: ...


@runtime_checkable
class TextLookup(Protocol):
    """Fetches the text for a document id (for the cross-encoder)."""

    def __call__(self, doc_id: ID) -> str: ...


@runtime_checkable
class Tokenizer(Protocol):
    """Splits text into tokens (for the BM25 index)."""

    def __call__(self, text: str) -> list[str]: ...


@runtime_checkable
class Normalize(Protocol):
    """Optional external score normalizer for weighted fusion."""

    def __call__(self, scores: Sequence[float], kind: str) -> Sequence[float]: ...


# --------------------------------------------------------------------------- #
# Configuration & result value objects (both frozen / immutable).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FusionProfile:
    """Tunable fusion + rerank configuration.

    ``weights`` defaults to an immutable mapping keyed by the canonical source
    constants; it must be immutable because a mutable default on a frozen
    dataclass would be a shared, corruptible object.
    """

    method: Literal["rrf", "weighted"] = "rrf"
    rrf_k: int = 60  # RRF damping constant
    weights: Mapping[str, float] = field(
        default_factory=lambda: MappingProxyType(
            {SOURCE_DENSE: 0.5, SOURCE_LEXICAL: 0.5}
        )
    )
    rerank_top_m: int = 50  # CE window; clamped up to k at query time
    ce_sigmoid: bool = False  # squash CE logits to [0, 1] for the gate hand-off


@dataclass(frozen=True)
class HybridResult:
    """The returned ranked result plus per-source provenance.

    ``scores`` are float64 and descending: cross-encoder scores when reranking,
    fused scores otherwise. ``provenance`` carries the ``dense``/``lexical``/
    ``fused``/``ce`` columns aligned to ``ids``, with NaN where a source did not
    surface a given id.
    """

    ids: tuple[ID, ...]
    scores: np.ndarray
    provenance: Mapping[str, np.ndarray]


# --------------------------------------------------------------------------- #
# Exceptions. All fusion-recall errors are ValueErrors so a host can catch the
# whole family with ``except FusionError`` or fall back to ``except ValueError``.
# --------------------------------------------------------------------------- #
class FusionError(ValueError):
    """Base class for all fusion-recall errors."""


class EmptyCorpusError(FusionError):
    """Raised when an operation requires a non-empty corpus but none exists."""


class InvalidProfileError(FusionError):
    """Raised at construction time for malformed / inconsistent configuration."""


# --------------------------------------------------------------------------- #
# Constants.
# --------------------------------------------------------------------------- #
DEFAULT_PROFILE: FusionProfile = FusionProfile()
DEFAULT_CE_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
