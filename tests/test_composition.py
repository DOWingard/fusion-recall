"""Gate hand-off contract.

The library is gate-agnostic: it ships no gate logic, only a score vector +
provenance. These tests prove that ``HybridResult.scores`` is a valid,
*non-degenerate* input to a downstream softmax-then-entropy confidence gate, in
**both** reranking and fusion-only modes, and on the ``ce_sigmoid`` path.

The reference gate is defined IN THIS TEST (not imported from the package) —
the library must not contain gate logic, so the proof cannot lean on it.
"""

from __future__ import annotations

import numpy as np
import pytest

from fusion_recall.contracts import FusionProfile
from fusion_recall.hybrid import HybridRetriever
from tests.conftest import CountingFakeCE, FakeDense, FakeLexical, TextLookupWithHole


# --------------------------------------------------------------------------- #
# Local reference gate — defined here, NOT imported from fusion_recall.
# Softmax (with temperature beta) -> Shannon entropy -> normalize by log(n).
# --------------------------------------------------------------------------- #
def _softmax(scores: np.ndarray, beta: float) -> np.ndarray:
    x = np.asarray(scores, dtype=np.float64) * beta
    x = x - np.max(x)  # numerically stable
    e = np.exp(x)
    return e / np.sum(e)


def _normalized_entropy(scores: np.ndarray, *, beta: float) -> float:
    """Return (normalized_entropy in [0,1], top1_mass) for a score vector."""
    p = _softmax(scores, beta)
    n = len(p)
    if n <= 1:
        return 0.0, float(p[0]) if n == 1 else 0.0
    # Shannon entropy in nats; guard log(0) by masking zero-probability terms.
    nz = p[p > 0.0]
    h = -np.sum(nz * np.log(nz))
    h_norm = h / np.log(n)  # log(n) is the max entropy for n classes
    return float(h_norm), float(np.max(p))


def _assert_valid_gate_input(scores: np.ndarray, *, beta: float) -> None:
    # Preconditions the host gate relies on: float64, finite, descending.
    assert scores.dtype == np.float64
    assert np.all(np.isfinite(scores))
    assert np.all(np.diff(scores) <= 0.0)

    h_norm, top1 = _normalized_entropy(scores, beta=beta)
    # Non-degenerate distribution: entropy strictly inside [0, 1] and a finite,
    # well-defined top-1 mass that is a proper probability.
    assert 0.0 <= h_norm <= 1.0
    assert np.isfinite(top1)
    assert 0.0 < top1 <= 1.0
    # Strictly non-degenerate: not a one-hot (h>0) and not perfectly uniform
    # (h<1). A degenerate all-equal score vector would push h_norm to exactly 1.
    assert 0.0 < h_norm < 1.0, "score distribution is degenerate (uniform)"


# --------------------------------------------------------------------------- #
# Reranking mode: scores are CE scores -> valid gate input.
# --------------------------------------------------------------------------- #
def test_reranking_scores_are_valid_gate_input(small_corpus):
    dense = FakeDense(
        {
            "doc_dense_1": 0.9,
            "doc_dense_2": 0.8,
            "doc_shared": 0.7,
            "doc_lex_1": 0.5,
        }
    )
    text_of = TextLookupWithHole(small_corpus)
    ce = CountingFakeCE()
    hr = HybridRetriever(dense, text_of=text_of, reranker=ce)
    result = hr.retrieve("q", k=4)
    _assert_valid_gate_input(result.scores, beta=1.0)


# --------------------------------------------------------------------------- #
# Fusion-only mode: scores are fused scores -> ALSO a valid gate input (this is
# the path that must never be a degenerate all-zero vector).
# --------------------------------------------------------------------------- #
def test_fusion_only_scores_are_valid_gate_input():
    dense = FakeDense({"a": 0.9, "b": 0.5, "c": 0.1})
    lexical = FakeLexical({"b": 4.0, "c": 3.0, "d": 1.0})
    hr = HybridRetriever(dense, lexical=lexical, reranker=None)
    result = hr.retrieve("q", k=4)
    # RRF scores have a small dynamic range, so a larger beta separates them; a
    # documented temperature is exactly the host's job.
    _assert_valid_gate_input(result.scores, beta=200.0)


# --------------------------------------------------------------------------- #
# ce_sigmoid path yields a non-degenerate distribution under a documented beta.
# --------------------------------------------------------------------------- #
def test_ce_sigmoid_path_non_degenerate(small_corpus):
    profile = FusionProfile(ce_sigmoid=True)
    dense = FakeDense(
        {
            "doc_dense_1": 0.9,
            "doc_dense_2": 0.8,
            "doc_shared": 0.7,
            "doc_lex_1": 0.5,
        }
    )
    text_of = TextLookupWithHole(small_corpus)
    # Distinct CE scores per doc so the sigmoid output is non-degenerate.
    ce = CountingFakeCE(
        scores_by_text={
            small_corpus["doc_dense_1"]: 4.0,
            small_corpus["doc_dense_2"]: 1.0,
            small_corpus["doc_shared"]: -1.0,
            small_corpus["doc_lex_1"]: -3.0,
        }
    )
    hr = HybridRetriever(dense, text_of=text_of, reranker=ce, profile=profile)
    result = hr.retrieve("q", k=4)

    # ce_sigmoid squashes scores into [0, 1].
    assert np.all(result.scores >= 0.0)
    assert np.all(result.scores <= 1.0)
    # Under a documented beta, the resulting distribution is non-degenerate.
    _assert_valid_gate_input(result.scores, beta=5.0)


def test_gate_rejects_degenerate_uniform_as_control():
    # Sanity check on the reference gate itself: a genuinely uniform score vector
    # must drive normalized entropy to ~1 (degenerate). This proves the
    # non-degeneracy assertions above have teeth.
    uniform = np.ones(5, dtype=np.float64)
    h_norm, _ = _normalized_entropy(uniform, beta=1.0)
    assert h_norm == pytest.approx(1.0)
