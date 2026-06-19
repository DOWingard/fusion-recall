"""Shared offline test doubles and fixtures for the fusion-recall suite.

Every double is a
deterministic, dependency-free stand-in for a seam Protocol so the entire core
suite runs with no model download and no I/O.

Each double genuinely satisfies its Protocol (``isinstance`` is True against the
runtime-checkable Protocol in ``fusion_recall.contracts``). Doubles are exposed
both as importable classes (for direct construction in tests that need bespoke
data) and as pytest fixtures (for the common cases).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pytest

from fusion_recall.contracts import ID, Ranking

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _as_ranking(data: Ranking | Mapping[ID, float]) -> Ranking:
    """Normalize static input into a descending Ranking.

    Accepts either an already-ordered ``Ranking`` (tuple of (id, score) pairs)
    or an ``id -> score`` mapping, which is sorted descending by score with a
    stable, deterministic tie-break on the string form of the id.
    """
    if isinstance(data, Mapping):
        pairs = list(data.items())
    else:
        pairs = list(data)
    pairs.sort(key=lambda kv: (-float(kv[1]), str(kv[0])))
    return tuple((doc_id, float(score)) for doc_id, score in pairs)


# --------------------------------------------------------------------------- #
# Dense retriever doubles
# --------------------------------------------------------------------------- #


class FakeDense:
    """Deterministic ``DenseRetriever`` built from static data.

    Returns a descending ``Ranking`` truncated to the requested ``k`` and
    records every ``(query, k)`` it was called with so callers can assert that
    ``candidate_k`` is applied per source. The same instance is fully
    deterministic across repeated calls (no internal mutation of the data).

    Construct a *lexical-blind* variant with :meth:`without`, which omits one or
    more ids — used to drive the recall-ceiling test (the dense retriever misses
    a document that only the lexical path can recover) and the eval dense
    baseline. Construct *structurally-distinct* variants (standing in for
    different host "systems") with :meth:`relabeled`, which composes a fresh,
    differently-shaped ranking.
    """

    def __init__(self, data: Ranking | Mapping[ID, float]) -> None:
        self._ranking: Ranking = _as_ranking(data)
        self.calls: list[tuple[str, int]] = []

    def __call__(self, query: str, k: int) -> Ranking:
        self.calls.append((query, k))
        return self._ranking[:k]

    @property
    def full_ranking(self) -> Ranking:
        """The complete underlying ranking (untruncated), for assertions."""
        return self._ranking

    def without(self, *doc_ids: ID) -> "FakeDense":
        """Return a new lexical-blind FakeDense that omits ``doc_ids``."""
        omit = set(doc_ids)
        kept = {doc_id: score for doc_id, score in self._ranking if doc_id not in omit}
        return FakeDense(kept)

    def relabeled(self, prefix: str) -> "FakeDense":
        """Return a structurally-distinct FakeDense with re-prefixed ids.

        Produces a ranking over a disjoint id space and a different score
        spread, so several of these stand in for distinct embedding "systems"
        without any of them exposing a vector.
        """
        relabeled = {
            f"{prefix}{doc_id}": score + 0.01 * (i + 1)
            for i, (doc_id, score) in enumerate(self._ranking)
        }
        return FakeDense(relabeled)


# --------------------------------------------------------------------------- #
# Lexical retriever doubles
# --------------------------------------------------------------------------- #


class FakeLexical:
    """Deterministic ``LexicalRetriever`` returning a fixed descending Ranking.

    Records every ``(query, k)`` it was called with (mirrors FakeDense) so the
    per-source ``candidate_k`` assertion can check the lexical side too.
    """

    def __init__(self, data: Ranking | Mapping[ID, float]) -> None:
        self._ranking: Ranking = _as_ranking(data)
        self.calls: list[tuple[str, int]] = []

    def __call__(self, query: str, k: int) -> Ranking:
        self.calls.append((query, k))
        return self._ranking[:k]

    @property
    def full_ranking(self) -> Ranking:
        return self._ranking


class RaisingLexical:
    """A ``LexicalRetriever`` that always raises, to drive fail-open tests.

    Records calls before raising so a test can assert it was actually invoked.
    """

    def __init__(self, exc: Exception | None = None) -> None:
        self._exc = exc or RuntimeError("lexical retrieval failed")
        self.calls: list[tuple[str, int]] = []

    def __call__(self, query: str, k: int) -> Ranking:
        self.calls.append((query, k))
        raise self._exc


# --------------------------------------------------------------------------- #
# Cross-encoder doubles
# --------------------------------------------------------------------------- #


class CountingFakeCE:
    """Deterministic ``CrossEncoder`` that counts invocations.

    ``score`` returns a deterministic ``np.ndarray`` of shape ``(len(docs),)``
    and float64 dtype. By default each document's score is the length of its
    text (a pure, general function of the input — not a hardcoded per-test
    table), which gives a non-trivial reordering signal distinct from any fused
    order. Pass ``scores_by_text`` to map specific document strings to specific
    scores when a test needs a precise ordering; texts absent from the table
    fall back to the length rule.

    ``calls`` counts invocations and ``doc_counts`` records ``len(docs)`` for
    each call, so tests can assert the CE is invoked exactly once on exactly
    ``min(top_m, len(candidates))`` documents (and never invoked in fusion-only
    mode).
    """

    def __init__(self, scores_by_text: Mapping[str, float] | None = None) -> None:
        self._table = dict(scores_by_text) if scores_by_text else {}
        self.calls: int = 0
        self.doc_counts: list[int] = []

    def score(self, query: str, docs: Sequence[str]) -> np.ndarray:
        self.calls += 1
        self.doc_counts.append(len(docs))
        values = [
            self._table[doc] if doc in self._table else float(len(doc))
            for doc in docs
        ]
        return np.asarray(values, dtype=np.float64)


class RaisingCE:
    """A ``CrossEncoder`` whose ``score`` always raises, to drive CE fail-open.

    Counts calls before raising so a test can assert it was invoked.
    """

    def __init__(self, exc: Exception | None = None) -> None:
        self._exc = exc or RuntimeError("cross-encoder scoring failed")
        self.calls: int = 0

    def score(self, query: str, docs: Sequence[str]) -> np.ndarray:
        self.calls += 1
        raise self._exc


# --------------------------------------------------------------------------- #
# Corpus / text fixtures
# --------------------------------------------------------------------------- #


class TextLookupWithHole:
    """A ``TextLookup`` over a corpus that is missing text for some ids.

    For a known id it returns the stored text; for an id with no text it returns
    the empty string (the chosen behavior for the missing-text edge: the
    candidate ranks low but never crashes). ``misses`` records the ids that fell
    through the hole.
    """

    def __init__(self, corpus: Mapping[ID, str]) -> None:
        self._corpus = dict(corpus)
        self.misses: list[ID] = []

    def __call__(self, doc_id: ID) -> str:
        if doc_id in self._corpus:
            return self._corpus[doc_id]
        self.misses.append(doc_id)
        return ""


# --------------------------------------------------------------------------- #
# Pytest fixtures (common cases). Bespoke-data tests construct the classes
# above directly.
# --------------------------------------------------------------------------- #


@pytest.fixture
def small_corpus() -> dict[str, str]:
    """A small id -> text corpus for hybrid/rerank tests.

    Includes an exact-identifier document (``doc_exact``) whose rare token
    (``zqxwv_token``) appears in no other document, so a lexical-blind dense
    fake can omit it and the lexical path must recover it.
    """
    return {
        "doc_dense_1": "the quick brown fox jumps over the lazy dog",
        "doc_dense_2": "a general overview of search and ranking systems",
        "doc_exact": "error code zqxwv_token raised by the parser module",
        "doc_lex_1": "lexical matching favors rare tokens and identifiers",
        "doc_shared": "shared document surfaced by both dense and lexical",
    }


@pytest.fixture
def fake_dense(small_corpus: dict[str, str]) -> FakeDense:
    """A dense fake over the small corpus that OMITS the exact-match doc.

    This is the lexical-blind baseline used by the recall-ceiling test: the
    exact-identifier document is deliberately absent from the dense ranking.
    """
    ranking = {
        "doc_dense_1": 0.9,
        "doc_dense_2": 0.8,
        "doc_shared": 0.7,
        "doc_lex_1": 0.3,
    }
    return FakeDense(ranking)


@pytest.fixture
def fake_lexical() -> FakeLexical:
    """A lexical fake that DOES surface the exact-match doc near the top."""
    ranking = {
        "doc_exact": 5.0,
        "doc_lex_1": 3.0,
        "doc_shared": 2.0,
    }
    return FakeLexical(ranking)


@pytest.fixture
def raising_lexical() -> RaisingLexical:
    return RaisingLexical()


@pytest.fixture
def counting_ce() -> CountingFakeCE:
    return CountingFakeCE()


@pytest.fixture
def raising_ce() -> RaisingCE:
    return RaisingCE()


@pytest.fixture
def text_of(small_corpus: dict[str, str]) -> TextLookupWithHole:
    """A ``TextLookup`` over the small corpus (no holes by default)."""
    return TextLookupWithHole(small_corpus)


@pytest.fixture
def text_of_with_hole(small_corpus: dict[str, str]) -> TextLookupWithHole:
    """A ``TextLookup`` over the small corpus minus one id (a deliberate hole)."""
    holed = {k: v for k, v in small_corpus.items() if k != "doc_shared"}
    return TextLookupWithHole(holed)
