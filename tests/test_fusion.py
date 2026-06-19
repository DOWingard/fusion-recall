"""Tests for rank fusion: ``rrf`` and ``weighted_fuse``.

These contracts (including the same-id edge) are written before the
implementation. The ``rrf`` expected values are derived independently from the
formula ``1/(k + rank)`` with a **1-based** rank per source (the arithmetic is
shown inline in :func:`test_rrf_hand_computed`), not copied from any output.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from fusion_recall.fusion import _minmax, rrf, weighted_fuse


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _as_dict(ranking):
    """Collapse a Ranking into an id -> score dict for value assertions."""
    return {doc_id: score for doc_id, score in ranking}


def _ids(ranking):
    return [doc_id for doc_id, _ in ranking]


def _is_descending(ranking) -> bool:
    scores = [score for _, score in ranking]
    return all(a >= b for a, b in zip(scores, scores[1:]))


# --------------------------------------------------------------------------- #
# rrf — hand-computed worked two-list example
# --------------------------------------------------------------------------- #
def test_rrf_hand_computed():
    """RRF on a worked two-list example with k=60, rank 1-based.

    dense:   A@rank1, B@rank2, C@rank3
    lexical: B@rank1, C@rank2, D@rank3

    Contribution per appearance is 1/(k + rank). Summing across the two
    sources (an id absent from a list contributes nothing):

        A = 1/(60+1)                       = 1/61
        B = 1/(60+2) + 1/(60+1)           = 1/62 + 1/61
        C = 1/(60+3) + 1/(60+2)           = 1/63 + 1/62
        D = 1/(60+3)                       = 1/63

    Decimal: A=0.0163934..., B=0.0325224..., C=0.0320020..., D=0.0158730...
    so the descending order is B > C > A > D.
    """
    rankings = {
        "dense": (("A", 0.9), ("B", 0.8), ("C", 0.7)),
        "lexical": (("B", 5.0), ("C", 3.0), ("D", 1.0)),
    }

    expected = {
        "A": 1 / 61,
        "B": 1 / 62 + 1 / 61,
        "C": 1 / 63 + 1 / 62,
        "D": 1 / 63,
    }

    out = rrf(rankings, k=60)
    got = _as_dict(out)

    assert set(got) == set(expected)
    for doc_id, want in expected.items():
        assert got[doc_id] == pytest.approx(want, rel=0, abs=1e-15)

    # Order is purely a consequence of the formula above.
    assert _ids(out) == ["B", "C", "A", "D"]
    assert _is_descending(out)


def test_rrf_default_k_is_60():
    """The default damping constant is 60 (matches explicit k=60)."""
    rankings = {
        "dense": (("A", 0.9), ("B", 0.8)),
        "lexical": (("B", 5.0), ("C", 3.0)),
    }
    assert rrf(rankings) == rrf(rankings, k=60)


def test_rrf_k_changes_contribution():
    """A different k changes the contribution magnitude per the formula.

    With a single one-list ranking [A@1, B@2] and k=10:
        A = 1/(10+1) = 1/11, B = 1/(10+2) = 1/12.
    """
    rankings = {"dense": (("A", 9.0), ("B", 1.0))}
    got = _as_dict(rrf(rankings, k=10))
    assert got["A"] == pytest.approx(1 / 11, abs=1e-15)
    assert got["B"] == pytest.approx(1 / 12, abs=1e-15)


# --------------------------------------------------------------------------- #
# rrf — set / union semantics
# --------------------------------------------------------------------------- #
def test_rrf_id_set_is_union():
    rankings = {
        "dense": (("A", 0.9), ("B", 0.8), ("C", 0.7)),
        "lexical": (("B", 5.0), ("C", 3.0), ("D", 1.0)),
    }
    out = rrf(rankings)
    assert set(_ids(out)) == {"A", "B", "C", "D"}


def test_rrf_absent_in_one_list_no_penalty():
    """An id absent from a source contributes nothing (no penalty, no error).

    A appears only in dense@rank1; D appears only in lexical@rank1. With k=60
    both get exactly one contribution of 1/(60+1) = 1/61 and therefore tie.
    """
    rankings = {
        "dense": (("A", 1.0),),
        "lexical": (("D", 1.0),),
    }
    got = _as_dict(rrf(rankings, k=60))
    assert got["A"] == pytest.approx(1 / 61, abs=1e-15)
    assert got["D"] == pytest.approx(1 / 61, abs=1e-15)
    assert got["A"] == got["D"]


def test_rrf_same_id_from_both_lists_single_entry():
    """Edge: an id in BOTH lists fuses to ONE entry whose score sums.

    X is dense@rank1 and lexical@rank1 → 1/(60+1) + 1/(60+1) = 2/61, a single
    row. Y appears only once (dense@rank2) → 1/(60+2).
    """
    rankings = {
        "dense": (("X", 0.9), ("Y", 0.8)),
        "lexical": (("X", 9.0),),
    }
    out = rrf(rankings, k=60)
    ids = _ids(out)
    assert ids.count("X") == 1  # single entry despite appearing in both lists
    got = _as_dict(out)
    assert got["X"] == pytest.approx(2 / 61, abs=1e-15)
    assert got["Y"] == pytest.approx(1 / 62, abs=1e-15)
    assert ids == ["X", "Y"]  # 2/61 > 1/62


# --------------------------------------------------------------------------- #
# rrf — commutativity over source order
# --------------------------------------------------------------------------- #
def test_rrf_commutative_over_source_order():
    """RRF is invariant to the order in which source lists are supplied."""
    dense = (("A", 0.9), ("B", 0.8), ("C", 0.7))
    lexical = (("B", 5.0), ("C", 3.0), ("D", 1.0))

    forward = rrf({"dense": dense, "lexical": lexical})
    reverse = rrf({"lexical": lexical, "dense": dense})

    assert forward == reverse


# --------------------------------------------------------------------------- #
# rrf — rank-based: invariant under monotonic rescaling (property test)
# --------------------------------------------------------------------------- #
@st.composite
def _rankings_with_monotone_rescale(draw):
    """Build a Ranking and a monotonically-rescaled twin with identical order.

    We draw distinct ids and STRICTLY DECREASING scores (so the rank order is
    unambiguous), then apply a strictly-increasing transform to the scores.
    Because the transform is monotonic and order is strict, the induced ranking
    (by rank position) is identical — which is exactly what RRF must be blind to.
    """
    n = draw(st.integers(min_value=1, max_value=8))
    ids = draw(
        st.lists(
            st.integers(min_value=0, max_value=999),
            min_size=n,
            max_size=n,
            unique=True,
        )
    )

    # Strictly decreasing original scores via positive gaps.
    gaps = draw(
        st.lists(
            st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
            min_size=n,
            max_size=n,
        )
    )
    scores = []
    running = 100.0
    for g in gaps:
        scores.append(running)
        running -= g
    original = tuple(zip(ids, scores))

    # Strictly-increasing transform parameters (a>0).
    a = draw(st.floats(min_value=0.01, max_value=50.0, allow_nan=False, allow_infinity=False))
    b = draw(st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False))
    rescaled = tuple((doc_id, a * s + b) for doc_id, s in original)

    return original, rescaled


@given(_rankings_with_monotone_rescale())
def test_rrf_invariant_under_monotonic_rescale(pair):
    """RRF depends only on rank, so any monotonic rescale leaves it unchanged."""
    original, rescaled = pair
    out_original = rrf({"src": original})
    out_rescaled = rrf({"src": rescaled})
    assert out_original == out_rescaled


# --------------------------------------------------------------------------- #
# rrf — empties & determinism
# --------------------------------------------------------------------------- #
def test_rrf_empty_inputs_give_empty_output():
    assert rrf({}) == ()
    assert rrf({"dense": (), "lexical": ()}) == ()


def test_rrf_deterministic():
    rankings = {
        "dense": (("A", 0.9), ("B", 0.8), ("C", 0.7)),
        "lexical": (("B", 5.0), ("C", 3.0), ("D", 1.0)),
    }
    assert rrf(rankings) == rrf(rankings)


def test_rrf_tie_break_is_deterministic():
    """Equal RRF scores get a stable, deterministic tie-break.

    A and B each appear once at rank 1 in separate sources → equal score 1/61.
    The tie-break must be deterministic across runs and independent of source
    order, so re-running and swapping sources yields the same id order.
    """
    r1 = rrf({"s1": (("A", 1.0),), "s2": (("B", 1.0),)})
    r2 = rrf({"s2": (("B", 1.0),), "s1": (("A", 1.0),)})
    assert r1 == r2
    assert _is_descending(r1)


# --------------------------------------------------------------------------- #
# _minmax — built-in normalizer (internal)
# --------------------------------------------------------------------------- #
def test_minmax_maps_to_unit_interval():
    out = _minmax([2.0, 6.0, 10.0], "dense")
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(0.5)
    assert out[2] == pytest.approx(1.0)


def test_minmax_zero_range_is_safe():
    """All-equal scores have zero range; output is finite (no div-by-zero)."""
    out = _minmax([3.0, 3.0, 3.0], "lexical")
    assert all(math.isfinite(v) for v in out)
    # Degenerate source maps to a constant (0.0) so it contributes nothing.
    assert all(v == 0.0 for v in out)


def test_minmax_empty():
    assert list(_minmax([], "dense")) == []


# --------------------------------------------------------------------------- #
# weighted_fuse — hand-computed linear combination
# --------------------------------------------------------------------------- #
def test_weighted_fuse_hand_computed():
    """Weighted fusion of two min-max-normalized sources, weights 0.5/0.5.

    dense:   A=10, B=6, C=2  -> min 2, max 10, range 8
        A=(10-2)/8=1.0  B=(6-2)/8=0.5  C=(2-2)/8=0.0
    lexical: B=4, C=3, D=1   -> min 1, max 4, range 3
        B=(4-1)/3=1.0   C=(3-1)/3=2/3  D=(1-1)/3=0.0

    Weighted sum (absent id contributes 0):
        A = 0.5*1.0 + 0.5*0       = 0.5
        B = 0.5*0.5 + 0.5*1.0     = 0.75
        C = 0.5*0.0 + 0.5*(2/3)   = 1/3
        D = 0.5*0   + 0.5*0.0     = 0.0
    Descending order: B > A > C > D.
    """
    rankings = {
        "dense": (("A", 10.0), ("B", 6.0), ("C", 2.0)),
        "lexical": (("B", 4.0), ("C", 3.0), ("D", 1.0)),
    }
    weights = {"dense": 0.5, "lexical": 0.5}

    expected = {
        "A": 0.5,
        "B": 0.75,
        "C": 1 / 3,
        "D": 0.0,
    }

    out = weighted_fuse(rankings, weights)
    got = _as_dict(out)

    assert set(got) == set(expected)
    for doc_id, want in expected.items():
        assert got[doc_id] == pytest.approx(want, abs=1e-15)
    assert _ids(out) == ["B", "A", "C", "D"]
    assert _is_descending(out)


def test_weighted_fuse_respects_weights():
    """Asymmetric weights change the linear combination accordingly.

    Same normalized values as above but weights dense=1.0, lexical=0.0 means
    the result is purely the dense normalization:
        A=1.0, B=0.5, C=0.0, D=0.0 (D absent in dense -> 0).
    """
    rankings = {
        "dense": (("A", 10.0), ("B", 6.0), ("C", 2.0)),
        "lexical": (("B", 4.0), ("C", 3.0), ("D", 1.0)),
    }
    got = _as_dict(weighted_fuse(rankings, {"dense": 1.0, "lexical": 0.0}))
    assert got["A"] == pytest.approx(1.0, abs=1e-15)
    assert got["B"] == pytest.approx(0.5, abs=1e-15)
    assert got["C"] == pytest.approx(0.0, abs=1e-15)
    assert got["D"] == pytest.approx(0.0, abs=1e-15)


def test_weighted_fuse_id_set_is_union():
    rankings = {
        "dense": (("A", 10.0), ("B", 6.0)),
        "lexical": (("C", 4.0), ("D", 1.0)),
    }
    out = weighted_fuse(rankings, {"dense": 0.5, "lexical": 0.5})
    assert set(_ids(out)) == {"A", "B", "C", "D"}


def test_weighted_fuse_same_id_from_both_lists_single_entry():
    """Edge for weighted: an id in both lists fuses to one summed row.

    dense:   X=10, Y=2  -> X=1.0, Y=0.0
    lexical: X=4        -> single element, zero range -> X=0.0
    weights 0.5/0.5:
        X = 0.5*1.0 + 0.5*0.0 = 0.5 (one entry)
        Y = 0.5*0.0           = 0.0
    """
    rankings = {
        "dense": (("X", 10.0), ("Y", 2.0)),
        "lexical": (("X", 4.0),),
    }
    out = weighted_fuse(rankings, {"dense": 0.5, "lexical": 0.5})
    ids = _ids(out)
    assert ids.count("X") == 1
    got = _as_dict(out)
    assert got["X"] == pytest.approx(0.5, abs=1e-15)
    assert got["Y"] == pytest.approx(0.0, abs=1e-15)
    assert ids == ["X", "Y"]


# --------------------------------------------------------------------------- #
# weighted_fuse — injected Normalize vs built-in
# --------------------------------------------------------------------------- #
def test_weighted_fuse_uses_injected_normalize():
    """An injected Normalize callable overrides the built-in min-max.

    The injected normalizer returns each score unchanged (identity), so with
    weights 0.5/0.5 the fused score is the raw weighted sum:
        dense A=10, B=6 ; lexical B=4, C=1
        A = 0.5*10            = 5.0
        B = 0.5*6 + 0.5*4     = 5.0
        C = 0.5*1             = 0.5
    A different (identity) normalization yields a different result than min-max,
    proving the injection actually took effect.
    """
    calls: list[str] = []

    def identity_normalize(scores, kind):
        calls.append(kind)
        return list(scores)

    rankings = {
        "dense": (("A", 10.0), ("B", 6.0)),
        "lexical": (("B", 4.0), ("C", 1.0)),
    }
    got = _as_dict(
        weighted_fuse(
            rankings, {"dense": 0.5, "lexical": 0.5}, normalize=identity_normalize
        )
    )
    assert got["A"] == pytest.approx(5.0, abs=1e-15)
    assert got["B"] == pytest.approx(5.0, abs=1e-15)
    assert got["C"] == pytest.approx(0.5, abs=1e-15)

    # The injected normalizer was actually invoked, once per source, with the
    # source name as `kind`.
    assert sorted(calls) == ["dense", "lexical"]


def test_weighted_fuse_builtin_differs_from_identity():
    """Built-in min-max gives a different answer than identity normalization.

    Guards that the default really is min-max, not a pass-through: with min-max
    the dense source A=10,B=6 normalizes to A=1.0,B=0.0, so A != raw 5.0.
    """
    rankings = {"dense": (("A", 10.0), ("B", 6.0))}
    got = _as_dict(weighted_fuse(rankings, {"dense": 1.0}))
    assert got["A"] == pytest.approx(1.0, abs=1e-15)
    assert got["B"] == pytest.approx(0.0, abs=1e-15)


# --------------------------------------------------------------------------- #
# weighted_fuse — empties, determinism, ordering
# --------------------------------------------------------------------------- #
def test_weighted_fuse_empty_inputs_give_empty_output():
    assert weighted_fuse({}, {}) == ()
    assert weighted_fuse({"dense": ()}, {"dense": 0.5}) == ()


def test_weighted_fuse_deterministic():
    rankings = {
        "dense": (("A", 10.0), ("B", 6.0), ("C", 2.0)),
        "lexical": (("B", 4.0), ("C", 3.0), ("D", 1.0)),
    }
    weights = {"dense": 0.5, "lexical": 0.5}
    assert weighted_fuse(rankings, weights) == weighted_fuse(rankings, weights)


def test_weighted_fuse_descending():
    rankings = {
        "dense": (("A", 10.0), ("B", 6.0), ("C", 2.0)),
        "lexical": (("B", 4.0), ("C", 3.0), ("D", 1.0)),
    }
    out = weighted_fuse(rankings, {"dense": 0.5, "lexical": 0.5})
    assert _is_descending(out)


def test_weighted_fuse_missing_weight_treated_as_zero():
    """A source with no entry in `weights` contributes nothing (weight 0).

    Only dense is weighted; lexical-only ids therefore score 0 but still appear.
    """
    rankings = {
        "dense": (("A", 10.0), ("B", 6.0)),
        "lexical": (("C", 4.0),),
    }
    out = weighted_fuse(rankings, {"dense": 1.0})
    got = _as_dict(out)
    assert got["A"] == pytest.approx(1.0, abs=1e-15)
    assert got["B"] == pytest.approx(0.0, abs=1e-15)
    assert got["C"] == pytest.approx(0.0, abs=1e-15)
    assert set(got) == {"A", "B", "C"}
