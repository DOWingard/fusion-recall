"""Rank fusion: merge per-source rankings into one ranked pool.

Two fusers are provided, both deterministic (bit-identical re-runs) and both
returning a single ``Ranking`` over the *union* of input ids:

* :func:`rrf` — Reciprocal Rank Fusion. Purely rank-based: each appearance of an
  id at 1-based ``rank`` in a source contributes ``1/(k + rank)``; contributions
  are summed across sources. Because it reads only rank, it is invariant to any
  monotonic rescaling of the input scores and to the order in which sources are
  supplied.
* :func:`weighted_fuse` — a weighted linear combination of per-source
  *normalized* scores. Each source is normalized (the injected ``Normalize`` or
  the built-in min-max :func:`_minmax`), multiplied by its weight, and summed; an
  id absent from a source contributes 0.

Both sort descending by fused score with a stable, deterministic tie-break so
identical inputs always yield identical output.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from fusion_recall.contracts import ID, Normalize, Ranking


# --------------------------------------------------------------------------- #
# Deterministic tie-break.
# --------------------------------------------------------------------------- #
def _sort_descending(scores: Mapping[ID, float]) -> Ranking:
    """Sort an ``id -> fused score`` mapping into a descending ``Ranking``.

    Ties are broken by the string form of the id, giving a stable, deterministic
    order that is independent of insertion order (and therefore of the order in
    which source lists were supplied).
    """
    items = sorted(scores.items(), key=lambda kv: (-kv[1], str(kv[0])))
    return tuple((doc_id, float(score)) for doc_id, score in items)


# --------------------------------------------------------------------------- #
# Reciprocal Rank Fusion.
# --------------------------------------------------------------------------- #
def rrf(rankings: Mapping[str, Ranking], *, k: int = 60) -> Ranking:
    """Fuse rankings by Reciprocal Rank Fusion.

    For each source, the item at 1-based position ``rank`` contributes
    ``1 / (k + rank)`` to its id's fused score; contributions are summed across
    all sources. The output is the union of all input ids, sorted descending by
    fused score with a deterministic tie-break.

    Rank-based by construction: the score values inside each source are never
    read, only their order, so the result is invariant under any monotonic
    rescaling of those scores and under source ordering.
    """
    fused: dict[ID, float] = {}
    for ranking in rankings.values():
        for position, (doc_id, _score) in enumerate(ranking):
            rank = position + 1  # 1-based rank within this source
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return _sort_descending(fused)


# --------------------------------------------------------------------------- #
# Built-in min-max normalizer (internal; Normalize-shaped, not exported).
# --------------------------------------------------------------------------- #
def _minmax(scores: Sequence[float], kind: str) -> list[float]:
    """Min-max normalize ``scores`` into ``[0, 1]``.

    The built-in default for :func:`weighted_fuse`. A degenerate source (empty
    or all-equal, i.e. zero range) maps to all-zeros so it stays finite and
    simply contributes nothing after weighting — never a division by zero.

    ``kind`` (the source name) is accepted to match the ``Normalize`` Protocol
    shape; the built-in does not need it.
    """
    values = [float(s) for s in scores]
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span == 0.0:
        return [0.0] * len(values)
    return [(v - lo) / span for v in values]


# --------------------------------------------------------------------------- #
# Weighted linear fusion.
# --------------------------------------------------------------------------- #
def weighted_fuse(
    rankings: Mapping[str, Ranking],
    weights: Mapping[str, float],
    *,
    normalize: Normalize | None = None,
) -> Ranking:
    """Fuse rankings by a weighted linear combination of normalized scores.

    Each source's scores are normalized (the injected ``normalize`` callable, or
    the built-in min-max :func:`_minmax` when ``None``), multiplied by that
    source's weight, and summed per id across sources. An id absent from a source
    contributes 0; a source missing from ``weights`` has weight 0. The output is
    the union of all input ids, sorted descending with a deterministic tie-break.
    """
    norm = normalize if normalize is not None else _minmax

    fused: dict[ID, float] = {}
    for source, ranking in rankings.items():
        weight = float(weights.get(source, 0.0))
        ids = [doc_id for doc_id, _ in ranking]
        raw = [score for _, score in ranking]
        normalized = norm(raw, source)
        for doc_id, value in zip(ids, normalized):
            fused[doc_id] = fused.get(doc_id, 0.0) + weight * float(value)
    return _sort_descending(fused)
