"""Hybrid orchestration: wire dense ∥ lexical recall → fuse → (rerank) → cut.

This is the wiring layer. It composes the four frozen modules (``contracts``,
``bm25``, ``fusion``, ``rerank``) into the public surface:

* :class:`HybridRetriever` — holds the seams and runs the pipeline per query.
* :func:`lift` — the one-call constructor that builds the batteries-included
  BM25 index and cross-encoder from a corpus.

Topology: dense and lexical run in parallel over the full corpus
(each capped at ``candidate_k``), their rankings are fused into one union pool,
the cross-encoder reorders a capped window of that pool (the only series stage),
and the top-``k`` is cut from the result.

Two safety properties are enforced here because only here are
``lexical``/``text_of``/``reranker``/``profile``/``k`` known together:

* **Fail-open**: a lexical fault degrades to dense-only order; a
  cross-encoder fault degrades to fused order. Neither raises into the read path.
* **Window clamp / no-truncation**: the rerank window is
  ``max(profile.rerank_top_m, k)`` so the returned top-``k`` is always drawn from
  one cross-encoder-scored, single-scale, descending window.

In fusion-only mode (``reranker=None``) the rerank stage is skipped entirely and
the returned scores are the *fused* scores — never a degenerate all-zero vector.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping

import numpy as np

from fusion_recall.bm25 import BM25Index
from fusion_recall.contracts import (
    DEFAULT_CE_MODEL,
    DEFAULT_PROFILE,
    ID,
    SOURCE_DENSE,
    SOURCE_LEXICAL,
    CrossEncoder,
    DenseRetriever,
    Fuser,
    FusionProfile,
    HybridResult,
    InvalidProfileError,
    LexicalRetriever,
    Ranking,
    TextLookup,
)
from fusion_recall.fusion import rrf, weighted_fuse
from fusion_recall.rerank import SentenceTransformerCrossEncoder, rerank

_log = logging.getLogger(__name__)

# Provenance column keys. The two recall sources reuse the canonical SOURCE_*
# constants; "fused" and "ce" are the two derived columns.
_PROV_FUSED = "fused"
_PROV_CE = "ce"


class HybridRetriever:
    """Hybrid retriever: dense ∥ lexical recall → fuse → (rerank) → cut to k.

    The dense seam is opaque and vector-free: it is only ever called as
    ``dense(query, candidate_k)`` and only its returned ``(id, score)`` ranking
    is read. The library never sees, stores, or inspects an embedding vector.

    ``reranker=None`` selects fusion-only mode: the cross-encoder stage is
    skipped and the returned scores are the fused scores. ``fuse=None`` selects
    the fuser from ``profile.method`` (``rrf`` or ``weighted_fuse``).
    """

    def __init__(
        self,
        dense: DenseRetriever,
        lexical: LexicalRetriever | None = None,
        text_of: TextLookup | None = None,
        reranker: CrossEncoder | None = None,
        profile: FusionProfile = DEFAULT_PROFILE,
        fuse: Fuser | None = None,
        depth_policy: Callable[[np.ndarray], int] | None = None,
    ) -> None:
        self._dense = dense
        self._lexical = lexical
        self._text_of = text_of
        self._reranker = reranker
        self._profile = profile
        self._depth_policy = depth_policy
        # Resolve the fuser once: an injected Fuser wins; otherwise it is chosen
        # from profile.method and bound to the profile's parameters.
        self._fuse: Fuser = fuse if fuse is not None else self._fuser_from_profile(profile)

    @property
    def profile(self) -> FusionProfile:
        return self._profile

    # ------------------------------------------------------------------ #
    # Fuser selection
    # ------------------------------------------------------------------ #
    @staticmethod
    def _fuser_from_profile(profile: FusionProfile) -> Fuser:
        """Bind the profile-selected fuser to its parameters as a Fuser."""
        if profile.method == "weighted":
            def _weighted(rankings: Mapping[str, Ranking]) -> Ranking:
                return weighted_fuse(rankings, profile.weights)

            return _weighted

        def _rrf(rankings: Mapping[str, Ranking]) -> Ranking:
            return rrf(rankings, k=profile.rrf_k)

        return _rrf

    # ------------------------------------------------------------------ #
    # Recall (fail-open per source)
    # ------------------------------------------------------------------ #
    def _recall_dense(self, query: str, candidate_k: int) -> Ranking:
        try:
            return tuple(self._dense(query, candidate_k))
        except Exception as exc:  # noqa: BLE001 — fail-open is the contract
            _log.warning("dense retrieval failed (%s); degrading to empty", exc)
            return ()

    def _recall_lexical(self, query: str, candidate_k: int) -> Ranking | None:
        """Run the lexical seam fail-open.

        Returns ``None`` when there is no lexical seam OR it faulted — in both
        cases fusion proceeds over dense alone (degrade to dense order), and the
        lexical provenance column is all-NaN.
        """
        if self._lexical is None:
            return None
        try:
            return tuple(self._lexical(query, candidate_k))
        except Exception as exc:  # noqa: BLE001 — fail-open is the contract
            _log.warning(
                "lexical retrieval failed (%s); degrading to dense-only order", exc
            )
            return None

    # ------------------------------------------------------------------ #
    # Query
    # ------------------------------------------------------------------ #
    def retrieve(self, query: str, k: int, *, candidate_k: int = 100) -> HybridResult:
        """Retrieve the top-``k`` for ``query``.

        ``candidate_k`` is passed to *both* the dense and lexical seams (per
        source). The rerank window is clamped to ``max(profile.rerank_top_m, k)``
        so the returned top-``k`` is single-scale and descending. When a
        ``depth_policy`` is set, the cross-encoder depth is
        ``min(max(depth_policy(fused_scores), k), window, len(fused))``.
        """
        dense_ranking = self._recall_dense(query, candidate_k)
        lexical_ranking = self._recall_lexical(query, candidate_k)

        # Assemble the per-source rankings keyed by the canonical constants. The
        # lexical key is only present when the lexical seam produced a ranking;
        # absent ⇒ fusion runs over dense alone (degrade to dense order).
        rankings: dict[str, Ranking] = {SOURCE_DENSE: dense_ranking}
        if lexical_ranking is not None:
            rankings[SOURCE_LEXICAL] = lexical_ranking

        fused: Ranking = self._fuse(rankings)

        if not fused:
            return self._empty_result()

        fused_scores = np.asarray([score for _, score in fused], dtype=np.float64)

        # Window clamp (no-truncation): the floor is k so the returned top-k is
        # always drawn from one cross-encoder-scored window; the cap is the fused
        # pool size.
        window = max(self._profile.rerank_top_m, k)
        if self._depth_policy is not None:
            raw_depth = int(self._depth_policy(fused_scores))
        else:
            raw_depth = window
        depth = min(max(raw_depth, k), window, len(fused))

        if self._reranker is not None:
            ranked = rerank(
                query,
                fused,
                text_of=self._require_text_of(),
                encoder=self._reranker,
                top_m=depth,
                sigmoid=self._profile.ce_sigmoid,
            )
            ce_scored = True
        else:
            # Fusion-only: skip the cross-encoder entirely and keep the fused
            # scores. The depth bound still applies so the candidate set matches
            # what the reranked path would have considered.
            ranked = fused[:depth]
            ce_scored = False

        final = ranked[:k]
        ids = tuple(doc_id for doc_id, _ in final)
        scores = np.asarray([score for _, score in final], dtype=np.float64)

        provenance = self._build_provenance(
            ids=ids,
            dense_ranking=dense_ranking,
            lexical_ranking=lexical_ranking,
            fused=fused,
            reranked=final if ce_scored else (),
        )
        return HybridResult(ids=ids, scores=scores, provenance=provenance)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _require_text_of(self) -> TextLookup:
        if self._text_of is None:
            # A reranker without a text source is a construction error; lift()
            # guards this, but a directly-built HybridRetriever could miss it.
            raise InvalidProfileError(
                "a reranker is configured but no text_of was provided to fetch "
                "candidate text for the cross-encoder"
            )
        return self._text_of

    def _empty_result(self) -> HybridResult:
        empty = np.asarray([], dtype=np.float64)
        provenance = {
            SOURCE_DENSE: empty,
            SOURCE_LEXICAL: empty,
            _PROV_FUSED: empty,
            _PROV_CE: empty,
        }
        return HybridResult(ids=(), scores=empty, provenance=provenance)

    @staticmethod
    def _build_provenance(
        *,
        ids: tuple[ID, ...],
        dense_ranking: Ranking,
        lexical_ranking: Ranking | None,
        fused: Ranking,
        reranked: Ranking,
    ) -> dict[str, np.ndarray]:
        """Build per-column score arrays aligned to ``ids``.

        Each column carries the source's score for an id, or NaN where the source
        did not surface that id. The ``ce`` column is populated only from
        genuine cross-encoder output (``reranked``); it is therefore all-NaN in
        fusion-only mode and NaN for any id the cross-encoder did not score.
        """
        dense_map = dict(dense_ranking)
        lexical_map = dict(lexical_ranking) if lexical_ranking is not None else {}
        fused_map = dict(fused)
        ce_map = dict(reranked)

        def column(source_map: Mapping[ID, float]) -> np.ndarray:
            return np.asarray(
                [source_map.get(doc_id, np.nan) for doc_id in ids],
                dtype=np.float64,
            )

        return {
            SOURCE_DENSE: column(dense_map),
            SOURCE_LEXICAL: column(lexical_map),
            _PROV_FUSED: column(fused_map),
            _PROV_CE: column(ce_map),
        }


def lift(
    dense: DenseRetriever,
    corpus: Mapping[ID, str] | None = None,
    text_of: TextLookup | None = None,
    *,
    rerank: bool = True,
    profile: FusionProfile = DEFAULT_PROFILE,
    ce_model: str = DEFAULT_CE_MODEL,
    lexical: LexicalRetriever | None = None,
    reranker: CrossEncoder | None = None,
) -> "HybridRetriever":
    """Lift a dense retriever to a full hybrid retriever in one call.

    Builds the batteries-included BM25 index from ``corpus`` (unless an explicit
    ``lexical`` is supplied) and the default cross-encoder (unless ``rerank`` is
    ``False`` or an explicit ``reranker`` is supplied). Fails fast at build time
    on inconsistent configuration rather than silently defaulting:

    * if ``lexical`` is ``None`` and ``corpus`` is ``None`` there is nothing to
      build BM25 from → :class:`InvalidProfileError`;
    * if ``rerank`` is ``True`` and neither ``corpus`` nor ``text_of`` is given
      there is no text source for the cross-encoder → :class:`InvalidProfileError`.

    ``rerank=False`` selects fusion-only mode by setting ``reranker=None`` — it
    does **not** install a zero-returning cross-encoder, so fusion-only returns
    fused scores, never zeros.
    """
    if lexical is None and corpus is None:
        raise InvalidProfileError(
            "no lexical source: pass a `corpus` to build the built-in BM25 index "
            "or inject a `lexical` retriever"
        )
    if rerank and corpus is None and text_of is None:
        raise InvalidProfileError(
            "rerank=True requires a text source: pass a `corpus` or `text_of` so "
            "the cross-encoder can fetch candidate text (or set rerank=False)"
        )

    # Build BM25 from the corpus when no lexical seam was injected.
    if lexical is None:
        # corpus is guaranteed non-None here by the guard above.
        lexical = BM25Index.fit(list(corpus), list(corpus.values()))

    # Derive a text lookup from the corpus when only a corpus was given.
    resolved_text_of = text_of
    if resolved_text_of is None and corpus is not None:
        resolved_text_of = corpus.__getitem__

    # Select the reranker: explicit override wins; else the default CE when
    # reranking; else None (fusion-only).
    if rerank:
        resolved_reranker = reranker if reranker is not None else SentenceTransformerCrossEncoder(ce_model)
    else:
        resolved_reranker = None

    return HybridRetriever(
        dense,
        lexical=lexical,
        text_of=resolved_text_of,
        reranker=resolved_reranker,
        profile=profile,
    )
