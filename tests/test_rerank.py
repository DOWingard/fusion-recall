"""Contracts for the cross-encoder rerank stage.

There is **no blend**: the cross-encoder score *replaces*
the fused score within the window, and ``rerank`` returns **only** that window
(length ``min(top_m, len(candidates))``). The caller (the hybrid layer) is what
clamps ``top_m >= k``; that clamp is not exercised here.

All tests except the explicitly-``slow`` real-model test run fully offline
against the deterministic doubles in ``tests/conftest.py``.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pytest

from fusion_recall.contracts import CrossEncoder, Ranking
from fusion_recall.rerank import (
    NoOpCrossEncoder,
    SentenceTransformerCrossEncoder,
    rerank,
)
from tests.conftest import CountingFakeCE, RaisingCE, TextLookupWithHole

# --------------------------------------------------------------------------- #
# Fixtures local to the rerank contracts.
# --------------------------------------------------------------------------- #


@pytest.fixture
def rerank_corpus() -> dict[str, str]:
    """A corpus whose texts have *strictly increasing* length by id.

    ``CountingFakeCE`` defaults to scoring each doc by ``len(text)``; choosing
    lengths that are the reverse of the fused order forces a real reordering so
    a no-op or identity bug is observable.
    """
    return {
        "a": "x",  # len 1
        "b": "xx",  # len 2
        "c": "xxx",  # len 3
        "d": "xxxx",  # len 4
        "e": "xxxxx",  # len 5
    }


@pytest.fixture
def fused() -> Ranking:
    """A fused ranking in descending fused-score order: a, b, c, d, e."""
    return (
        ("a", 5.0),
        ("b", 4.0),
        ("c", 3.0),
        ("d", 2.0),
        ("e", 1.0),
    )


def _text_of(corpus: dict[str, str]) -> TextLookupWithHole:
    return TextLookupWithHole(corpus)


# --------------------------------------------------------------------------- #
# Reordering within the window + CE replaces (no blend).
# --------------------------------------------------------------------------- #


def test_rerank_reorders_window_by_ce_score(rerank_corpus, fused):
    """CE score (len(text)) reverses the fused order within the window."""
    ce = CountingFakeCE()
    out = rerank(
        "q",
        fused,
        text_of=_text_of(rerank_corpus),
        encoder=ce,
        top_m=3,
    )
    # Window is the first 3 fused ids {a,b,c}; CE scores are their lengths
    # (1, 2, 3) so descending CE order is c, b, a.
    assert [doc_id for doc_id, _ in out] == ["c", "b", "a"]


def test_rerank_replaces_scores_with_ce_no_blend(rerank_corpus, fused):
    """The returned scores ARE the CE scores, not the fused scores or a blend."""
    ce = CountingFakeCE()
    out = rerank(
        "q",
        fused,
        text_of=_text_of(rerank_corpus),
        encoder=ce,
        top_m=3,
    )
    scores = {doc_id: score for doc_id, score in out}
    # len("xxx")=3, len("xx")=2, len("x")=1 — exactly the CE outputs, untouched
    # by the fused scores 3.0/4.0/5.0.
    assert scores == {"c": 3.0, "b": 2.0, "a": 1.0}


def test_rerank_returns_only_the_window_no_remainder(rerank_corpus, fused):
    """Return length is min(top_m, len); ids outside the window are absent."""
    ce = CountingFakeCE()
    out = rerank(
        "q",
        fused,
        text_of=_text_of(rerank_corpus),
        encoder=ce,
        top_m=2,
    )
    assert len(out) == 2
    returned_ids = {doc_id for doc_id, _ in out}
    # d and e were outside the top-2 window and must NOT be appended.
    assert returned_ids == {"a", "b"}
    assert "c" not in returned_ids
    assert "d" not in returned_ids
    assert "e" not in returned_ids


def test_rerank_top_m_exceeds_len_returns_whole_list(rerank_corpus, fused):
    """min(top_m, len) clamps the window to the candidate count, not top_m."""
    ce = CountingFakeCE()
    out = rerank(
        "q",
        fused,
        text_of=_text_of(rerank_corpus),
        encoder=ce,
        top_m=999,
    )
    assert len(out) == len(fused)
    # Full reorder by length: e(5), d(4), c(3), b(2), a(1).
    assert [doc_id for doc_id, _ in out] == ["e", "d", "c", "b", "a"]


# --------------------------------------------------------------------------- #
# CE invoked on exactly min(top_m, len) documents.
# --------------------------------------------------------------------------- #


def test_ce_invoked_on_exactly_window_size(rerank_corpus, fused):
    ce = CountingFakeCE()
    rerank("q", fused, text_of=_text_of(rerank_corpus), encoder=ce, top_m=3)
    assert ce.calls == 1
    assert ce.doc_counts == [3]


def test_ce_invoked_on_clamped_window_size_when_top_m_large(rerank_corpus, fused):
    ce = CountingFakeCE()
    rerank("q", fused, text_of=_text_of(rerank_corpus), encoder=ce, top_m=100)
    assert ce.calls == 1
    assert ce.doc_counts == [len(fused)]


# --------------------------------------------------------------------------- #
# sigmoid path.
# --------------------------------------------------------------------------- #


def test_sigmoid_squashes_scores_into_unit_interval(rerank_corpus, fused):
    ce = CountingFakeCE()
    out = rerank(
        "q",
        fused,
        text_of=_text_of(rerank_corpus),
        encoder=ce,
        top_m=5,
        sigmoid=True,
    )
    for _, score in out:
        assert 0.0 <= score <= 1.0


def test_sigmoid_matches_logistic_of_raw_ce(rerank_corpus, fused):
    """The squashed score is exactly 1/(1+exp(-raw)) of the CE logit."""
    ce = CountingFakeCE()
    out = rerank(
        "q",
        fused,
        text_of=_text_of(rerank_corpus),
        encoder=ce,
        top_m=5,
        sigmoid=True,
    )
    scores = {doc_id: score for doc_id, score in out}
    # Raw CE for "a" is len("x") == 1.0.
    assert scores["a"] == pytest.approx(1.0 / (1.0 + np.exp(-1.0)))
    # Raw CE for "e" is len("xxxxx") == 5.0.
    assert scores["e"] == pytest.approx(1.0 / (1.0 + np.exp(-5.0)))


# --------------------------------------------------------------------------- #
# NoOpCrossEncoder: window order unchanged (zeros under a stable sort).
# --------------------------------------------------------------------------- #


def test_noop_leaves_window_order_unchanged(rerank_corpus, fused):
    out = rerank(
        "q",
        fused,
        text_of=_text_of(rerank_corpus),
        encoder=NoOpCrossEncoder(),
        top_m=4,
    )
    # All-zero CE scores cannot reorder a stable sort, so fused order is kept,
    # truncated to the window.
    assert [doc_id for doc_id, _ in out] == ["a", "b", "c", "d"]
    for _, score in out:
        assert score == 0.0


def test_noop_score_returns_float64_zeros():
    docs = ["one", "two", "three"]
    arr = NoOpCrossEncoder().score("q", docs)
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float64
    assert arr.shape == (len(docs),)
    assert np.all(arr == 0.0)


# --------------------------------------------------------------------------- #
# Fail-open: a CE that raises degrades to the input order + a logged warning.
# --------------------------------------------------------------------------- #


def test_ce_raises_fails_open_to_input_order(rerank_corpus, fused, caplog):
    ce = RaisingCE()
    with caplog.at_level(logging.WARNING):
        out = rerank(
            "q",
            fused,
            text_of=_text_of(rerank_corpus),
            encoder=ce,
            top_m=3,
        )
    # No exception propagated and the original input is returned unchanged.
    assert out == fused
    assert ce.calls == 1


def test_ce_raises_logs_a_warning(rerank_corpus, fused, caplog):
    ce = RaisingCE()
    with caplog.at_level(logging.WARNING):
        rerank("q", fused, text_of=_text_of(rerank_corpus), encoder=ce, top_m=3)
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


# --------------------------------------------------------------------------- #
# Protocol conformance.
# --------------------------------------------------------------------------- #


def test_noop_satisfies_crossencoder_protocol():
    assert isinstance(NoOpCrossEncoder(), CrossEncoder)


def test_sentence_transformer_satisfies_crossencoder_protocol():
    # Construction must NOT import torch / load a model; the Protocol check is
    # purely structural (does it have a conforming ``score`` method).
    st = SentenceTransformerCrossEncoder()
    assert isinstance(st, CrossEncoder)


# --------------------------------------------------------------------------- #
# Slow / optional: the real model. Must SKIP (never hang or error) when the
# model is not available offline.
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_real_model_scores_relevant_above_irrelevant():
    pytest.importorskip("sentence_transformers")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        st = SentenceTransformerCrossEncoder()
        scores = st.score(
            "how do I sort a list in python",
            [
                "Use the built-in sorted() function to sort a python list.",
                "The mitochondria is the powerhouse of the cell.",
            ],
        )
    except Exception as exc:  # model not cached / unavailable offline
        pytest.skip(f"real cross-encoder model unavailable: {exc}")
    assert scores[0] > scores[1]
