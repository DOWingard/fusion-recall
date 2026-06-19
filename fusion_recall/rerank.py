"""Cross-encoder rerank stage: the single series, precision-oriented stage.

The cross-encoder re-scores each ``(query, document)`` pair with full
cross-attention and is the precision engine of the pipeline. It is always
capped to a window of candidates because it is too expensive for the full
corpus.

There is **no blend**: within the window the cross-encoder
score *replaces* the fused score, and :func:`rerank` returns **only** that
window. Clamping the window up to ``k`` (so the returned top-``k`` is
single-scale) is the caller's responsibility, not this module's.

Any cross-encoder fault degrades to the input order with a logged warning and
never raises into the host's read path (fail-open).

The default :class:`SentenceTransformerCrossEncoder` imports
``sentence_transformers`` lazily and behind an ``ImportError`` guard so that
importing this module never pulls in torch.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from fusion_recall.contracts import (
    DEFAULT_CE_MODEL,
    CrossEncoder,
    Ranking,
    TextLookup,
)

_log = logging.getLogger(__name__)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Numerically-stable logistic squash to ``[0, 1]``."""
    return 1.0 / (1.0 + np.exp(-values))


def rerank(
    query: str,
    candidates: Ranking,
    *,
    text_of: TextLookup,
    encoder: CrossEncoder,
    top_m: int,
    sigmoid: bool = False,
) -> Ranking:
    """Rerank the top-``top_m`` window of ``candidates`` by cross-encoder score.

    The window is the first ``min(top_m, len(candidates))`` candidates. Each
    windowed document's text is fetched via ``text_of`` and scored by
    ``encoder``; the cross-encoder score *replaces* the fused score (there is no
    blend). The window is re-sorted descending by that score and returned on its
    own — candidates outside the window are not appended here (the caller clamps
    ``top_m >= k`` so the returned window already covers the final top-``k``).

    Fail-open: any exception while fetching text or scoring is logged as a
    warning and the original ``candidates`` are returned unchanged, so a
    cross-encoder fault never propagates into the host's read path.
    """
    window_size = min(top_m, len(candidates))
    if window_size <= 0:
        return ()
    window = candidates[:window_size]

    try:
        window_texts = [text_of(doc_id) for doc_id, _ in window]
        scores = encoder.score(query, window_texts)
    except Exception as exc:  # noqa: BLE001 — fail-open is the contract
        _log.warning(
            "cross-encoder rerank failed (%s); falling back to input order", exc
        )
        return candidates

    scores = np.asarray(scores, dtype=np.float64)
    if sigmoid:
        scores = _sigmoid(scores)

    # Stable descending sort: equal CE scores preserve the incoming fused order,
    # which is why an all-zero (NoOp) score vector leaves the window untouched.
    order = sorted(
        range(window_size), key=lambda i: float(scores[i]), reverse=True
    )
    return tuple((window[i][0], float(scores[i])) for i in order)


class NoOpCrossEncoder:
    """Identity cross-encoder: scores every document ``0.0``.

    Equal scores cannot reorder a stable sort, so passing this to
    :func:`rerank` leaves the window in its incoming fused order. It is an
    explicit, dependency-free stand-in; the hybrid layer's fusion-only mode uses
    ``reranker=None`` and skips reranking entirely rather than routing through
    this class.
    """

    def score(self, query: str, docs: Sequence[str]) -> np.ndarray:
        return np.zeros(len(docs), dtype=np.float64)


class SentenceTransformerCrossEncoder:
    """Default cross-encoder backed by ``sentence_transformers.CrossEncoder``.

    The heavy ``sentence_transformers`` import (which pulls in torch) is
    deferred to the first :meth:`score` call and guarded by an ``ImportError``
    handler, so merely importing this module — or constructing this class — never
    triggers torch. Offline / cached operation is delegated to the
    ``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE`` environment variables, which a
    host sets after pre-caching the model.
    """

    def __init__(
        self,
        model: str = DEFAULT_CE_MODEL,
        *,
        device: str = "cpu",
        max_length: int = 512,
    ) -> None:
        self._model_name = model
        self._device = device
        self._max_length = max_length
        self._model = None  # lazily constructed on first score()

    def _ensure_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder as _STCrossEncoder
            except ImportError as exc:  # pragma: no cover - env-dependent
                raise ImportError(
                    "SentenceTransformerCrossEncoder requires the "
                    "'sentence-transformers' package; install it (and a CPU or "
                    "CUDA torch build) to use the default cross-encoder."
                ) from exc
            self._model = _STCrossEncoder(
                self._model_name,
                device=self._device,
                max_length=self._max_length,
            )
        return self._model

    def score(self, query: str, docs: Sequence[str]) -> np.ndarray:
        model = self._ensure_model()
        pairs = [(query, doc) for doc in docs]
        scores = model.predict(pairs)
        return np.asarray(scores, dtype=np.float64)
