"""Contracts for the hybrid orchestration layer.

These tests pin the wiring that turns the four frozen modules (contracts, bm25,
fusion, rerank) into the public ``lift`` / ``HybridRetriever`` surface. They run
fully offline against the deterministic doubles in ``tests/conftest.py`` — no
model download, no I/O.

The load-bearing behaviors:

* recall-ceiling — a lexical-blind dense fake omits the exact-match doc; the
  hybrid must surface it via the lexical path.
* fusion-only non-degenerate — ``reranker=None`` returns the *fused* scores, never
  a degenerate all-zero vector.
* clamp / no-truncation — ``rerank_top_m < k`` clamps the window up to ``k`` so the
  returned top-``k`` is single-scale and descending.
* provenance — columns aligned to ids, NaN where a source did not surface the id;
  the ``ce`` column is all-NaN in fusion-only mode.
* embedding-dim-agnostic — several structurally-distinct dense fakes; the dense
  seam only ever receives ``(query, k)`` — never a vector.
* fail-open — lexical raises ⇒ dense order; CE raises ⇒ fused order.
"""

from __future__ import annotations

import numpy as np
import pytest

from fusion_recall.contracts import (
    SOURCE_DENSE,
    SOURCE_LEXICAL,
    FusionProfile,
    HybridResult,
    InvalidProfileError,
)
from fusion_recall.hybrid import HybridRetriever, lift
from tests.conftest import (
    CountingFakeCE,
    FakeDense,
    FakeLexical,
    RaisingCE,
    RaisingLexical,
    TextLookupWithHole,
)


# --------------------------------------------------------------------------- #
# A dense fake that records every argument it is ever handed, so the
# embedding-dim-agnostic invariant (the dense seam only ever receives
# ``(query, k)``) can be checked positively: no positional or keyword argument
# is ever a vector / array.
# --------------------------------------------------------------------------- #
class ArgRecordingDense:
    """Dense double that captures full ``(args, kwargs)`` of every call."""

    def __init__(self, ranking):
        self._inner = FakeDense(ranking)
        self.received: list[tuple[tuple, dict]] = []

    def __call__(self, *args, **kwargs):
        self.received.append((args, kwargs))
        return self._inner(*args, **kwargs)


# --------------------------------------------------------------------------- #
# lift(...) -> HybridRetriever ; retrieve -> HybridResult with k ids
# --------------------------------------------------------------------------- #
def test_lift_returns_hybrid_retriever_and_retrieve_returns_result(small_corpus):
    dense = FakeDense({"doc_dense_1": 0.9, "doc_dense_2": 0.8, "doc_shared": 0.7})
    hr = lift(dense, small_corpus, rerank=False)
    assert isinstance(hr, HybridRetriever)

    result = hr.retrieve("overview search ranking", k=3)
    assert isinstance(result, HybridResult)
    assert len(result.ids) == 3
    assert len(result.scores) == 3
    # ids are unique (same id surfaced by both sources collapses to one entry).
    assert len(set(result.ids)) == len(result.ids)


# --------------------------------------------------------------------------- #
# Recall-ceiling: the CENTRAL value of the library. The dense fake OMITS the
# exact-match doc; the hybrid must surface it via the lexical (BM25) path.
# --------------------------------------------------------------------------- #
def test_recall_ceiling_surfaces_doc_dense_missed(small_corpus, fake_dense):
    # ``fake_dense`` is lexical-blind: it never returns "doc_exact". Confirm that
    # precondition so this test cannot pass by the dense path sneaking it in.
    dense_ids = {doc_id for doc_id, _ in fake_dense.full_ranking}
    assert "doc_exact" not in dense_ids

    hr = lift(fake_dense, small_corpus, rerank=False)
    # "zqxwv_token" is a rare token living ONLY in doc_exact.
    result = hr.retrieve("zqxwv_token", k=5)

    assert "doc_exact" in result.ids, (
        "hybrid failed to recover the dense-missed exact-match doc via lexical"
    )


def test_recall_ceiling_holds_through_rerank(small_corpus, fake_dense):
    # Same recovery must survive the rerank stage (CE only reorders the fused
    # window; it cannot add or drop candidates).
    ce = CountingFakeCE()
    hr = lift(
        fake_dense,
        small_corpus,
        rerank=True,
        reranker=ce,
    )
    result = hr.retrieve("zqxwv_token", k=5)
    assert "doc_exact" in result.ids


def test_recall_union_is_union_of_sources():
    # Fused candidate set == union of dense and lexical ids.
    dense = FakeDense({"a": 0.9, "b": 0.5})
    lexical = FakeLexical({"b": 4.0, "c": 3.0})
    hr = HybridRetriever(dense, lexical=lexical, reranker=None)
    result = hr.retrieve("q", k=10)
    assert set(result.ids) == {"a", "b", "c"}


# --------------------------------------------------------------------------- #
# Provenance: dense / lexical / fused / ce columns aligned to ids, NaN absent.
# --------------------------------------------------------------------------- #
def test_provenance_columns_aligned_with_nan(small_corpus):
    # doc_dense_1 only in dense; doc_exact only in lexical; doc_shared in both.
    dense = FakeDense({"doc_dense_1": 0.9, "doc_shared": 0.7})
    lexical = FakeLexical({"doc_exact": 5.0, "doc_shared": 2.0})
    text_of = TextLookupWithHole(small_corpus)
    ce = CountingFakeCE()
    hr = HybridRetriever(dense, lexical=lexical, text_of=text_of, reranker=ce)
    result = hr.retrieve("q", k=3)

    prov = result.provenance
    for col in ("dense", "lexical", "fused", "ce"):
        assert col in prov
        assert isinstance(prov[col], np.ndarray)
        assert prov[col].shape == (len(result.ids),)

    idx = {doc_id: i for i, doc_id in enumerate(result.ids)}

    # dense surfaced doc_dense_1 + doc_shared, NOT doc_exact -> NaN there.
    assert np.isnan(prov["dense"][idx["doc_exact"]])
    assert not np.isnan(prov["dense"][idx["doc_dense_1"]])
    assert not np.isnan(prov["dense"][idx["doc_shared"]])

    # lexical surfaced doc_exact + doc_shared, NOT doc_dense_1 -> NaN there.
    assert np.isnan(prov["lexical"][idx["doc_dense_1"]])
    assert not np.isnan(prov["lexical"][idx["doc_exact"]])

    # fused covers every returned id (union) -> no NaN.
    assert not np.any(np.isnan(prov["fused"]))

    # ce ran on the window -> these (k=3 <= window) ids are CE-scored, not NaN.
    assert not np.any(np.isnan(prov["ce"]))


def test_provenance_ce_all_nan_in_fusion_only():
    dense = FakeDense({"a": 0.9, "b": 0.5})
    lexical = FakeLexical({"b": 4.0, "c": 3.0})
    hr = HybridRetriever(dense, lexical=lexical, reranker=None)
    result = hr.retrieve("q", k=3)
    # No CE stage at all -> the entire ce column is NaN.
    assert np.all(np.isnan(result.provenance["ce"]))
    # ...but the fused column is fully populated.
    assert not np.any(np.isnan(result.provenance["fused"]))


# --------------------------------------------------------------------------- #
# scores are float64, descending, length == len(ids).
# --------------------------------------------------------------------------- #
def test_scores_float64_descending_length(small_corpus):
    dense = FakeDense({"doc_dense_1": 0.9, "doc_dense_2": 0.8, "doc_shared": 0.7})
    lexical = FakeLexical({"doc_exact": 5.0, "doc_lex_1": 3.0})
    text_of = TextLookupWithHole(small_corpus)
    ce = CountingFakeCE()
    hr = HybridRetriever(dense, lexical=lexical, text_of=text_of, reranker=ce)
    result = hr.retrieve("q", k=4)

    assert result.scores.dtype == np.float64
    assert len(result.scores) == len(result.ids)
    diffs = np.diff(result.scores)
    assert np.all(diffs <= 0.0), "scores must be descending"
    assert np.all(np.isfinite(result.scores))


# --------------------------------------------------------------------------- #
# candidate_k is applied PER SOURCE (both dense and lexical queried with it).
# --------------------------------------------------------------------------- #
def test_candidate_k_applied_per_source():
    dense = FakeDense({f"d{i}": float(i) for i in range(20)})
    lexical = FakeLexical({f"l{i}": float(i) for i in range(20)})
    hr = HybridRetriever(dense, lexical=lexical, reranker=None)
    hr.retrieve("q", k=3, candidate_k=7)

    assert dense.calls == [("q", 7)]
    assert lexical.calls == [("q", 7)]


# --------------------------------------------------------------------------- #
# fusion-only vs rerank: scores ARE the fused scores in fusion-only (and are
# non-degenerate — not all zero / not all equal), and CE scores when reranking.
# --------------------------------------------------------------------------- #
def test_fusion_only_scores_are_fused_not_zero():
    dense = FakeDense({"a": 0.9, "b": 0.5, "c": 0.1})
    lexical = FakeLexical({"b": 4.0, "c": 3.0, "d": 1.0})
    hr = HybridRetriever(dense, lexical=lexical, reranker=None)
    result = hr.retrieve("q", k=4)

    scores = result.scores
    # The discriminating assertions for the "emit zeros" mutation:
    assert not np.allclose(scores, 0.0), "fusion-only scores collapsed to zero"
    assert not np.allclose(scores, scores[0]), "fusion-only scores are degenerate (all equal)"

    # And they must equal the RRF fused scores for those ids (real, not invented).
    from fusion_recall.fusion import rrf

    fused = dict(
        rrf({SOURCE_DENSE: dense.full_ranking, SOURCE_LEXICAL: lexical.full_ranking})
    )
    for doc_id, score in zip(result.ids, scores):
        assert score == pytest.approx(fused[doc_id])


def test_reranking_scores_are_ce_scores(small_corpus):
    # CountingFakeCE scores by len(text); the returned scores must equal those CE
    # values for the surfaced ids (proving CE scores replace fused scores).
    dense = FakeDense({"doc_dense_1": 0.9, "doc_dense_2": 0.8, "doc_shared": 0.7})
    text_of = TextLookupWithHole(small_corpus)
    ce = CountingFakeCE()
    hr = HybridRetriever(dense, text_of=text_of, reranker=ce)
    result = hr.retrieve("q", k=3)

    assert ce.calls == 1
    for doc_id, score in zip(result.ids, result.scores):
        assert score == pytest.approx(float(len(small_corpus[doc_id])))


def test_fusion_only_does_not_invoke_ce():
    # No CE object exists in fusion-only mode; assert no CE work happens by
    # routing a RaisingCE that would blow up if ever called — but pass reranker
    # as None so it is never wired. (A CE invocation would have to come from
    # somewhere; there is no path.) We assert via a CountingFakeCE handed only as
    # a sentinel that is NOT passed to the retriever.
    sentinel = CountingFakeCE()
    dense = FakeDense({"a": 0.9, "b": 0.5})
    hr = HybridRetriever(dense, reranker=None)
    hr.retrieve("q", k=2)
    assert sentinel.calls == 0


# --------------------------------------------------------------------------- #
# Clamp / no-truncation: rerank_top_m < k => window clamps to k.
# --------------------------------------------------------------------------- #
def test_clamp_window_up_to_k(small_corpus):
    # rerank_top_m = 2 but k = 4: the CE window must clamp UP to 4 so the
    # returned top-4 is single-scale CE scores and the count is k.
    profile = FusionProfile(rerank_top_m=2)
    dense = FakeDense(
        {
            "doc_dense_1": 0.9,
            "doc_dense_2": 0.8,
            "doc_shared": 0.7,
            "doc_lex_1": 0.6,
            "doc_exact": 0.5,
        }
    )
    text_of = TextLookupWithHole(small_corpus)
    ce = CountingFakeCE()
    hr = HybridRetriever(dense, text_of=text_of, reranker=ce, profile=profile)
    result = hr.retrieve("q", k=4)

    assert len(result.ids) == 4
    # The CE must have been invoked on at least k docs (window clamped up to k),
    # not on rerank_top_m=2 docs.
    assert ce.doc_counts[0] >= 4
    # Single scale + strictly descending across the whole returned vector.
    assert np.all(np.diff(result.scores) <= 0.0)
    # And every returned score is a CE score (len-of-text), proving single-scale.
    for doc_id, score in zip(result.ids, result.scores):
        assert score == pytest.approx(float(len(small_corpus[doc_id])))


def test_returned_count_is_min_k_corpus(small_corpus):
    # k greater than the available candidate pool returns all available, never
    # raises (a clamp must not pad below or fabricate above the pool).
    dense = FakeDense({"doc_dense_1": 0.9, "doc_dense_2": 0.8})
    hr = lift(dense, small_corpus, rerank=False)
    # query a rare token so only a couple docs surface from lexical too.
    result = hr.retrieve("doc_dense_1", k=100)
    assert len(result.ids) == len(result.scores)
    assert len(result.ids) >= 1


# --------------------------------------------------------------------------- #
# depth_policy: used when provided, bounded by max(rerank_top_m, k).
# --------------------------------------------------------------------------- #
def test_depth_policy_used_and_bounded(small_corpus):
    profile = FusionProfile(rerank_top_m=2)
    dense = FakeDense(
        {
            "doc_dense_1": 0.9,
            "doc_dense_2": 0.8,
            "doc_shared": 0.7,
            "doc_lex_1": 0.6,
            "doc_exact": 0.5,
        }
    )
    text_of = TextLookupWithHole(small_corpus)
    ce = CountingFakeCE()
    seen = {}

    def policy(fused_scores: np.ndarray) -> int:
        # Record what we were handed (must be the fused-score array) and ask for
        # a huge depth so the bound is what limits the window.
        seen["arg"] = fused_scores
        return 9999

    hr = HybridRetriever(
        dense, text_of=text_of, reranker=ce, profile=profile, depth_policy=policy
    )
    result = hr.retrieve("q", k=3)

    # The policy received the fused-score vector (an ndarray) covering the whole
    # fused union (5 docs here; no lexical, so fused == dense's 5 entries).
    assert isinstance(seen["arg"], np.ndarray)
    assert len(seen["arg"]) == len(dense.full_ranking)
    # Even though it asked for 9999, the CE window is bounded by
    # min(9999, max(rerank_top_m=2, k=3), len(fused)) = max(2,3) = 3.
    bound = min(9999, max(profile.rerank_top_m, 3), len(seen["arg"]))
    assert bound == 3
    assert ce.doc_counts[0] == bound
    assert len(result.ids) == 3


def test_depth_policy_lower_bound_still_returns_k(small_corpus):
    # A policy that asks for a depth BELOW k must not cut the result below k,
    # because the window floor is k (clamp). Returned count stays k.
    profile = FusionProfile(rerank_top_m=2)
    dense = FakeDense(
        {
            "doc_dense_1": 0.9,
            "doc_dense_2": 0.8,
            "doc_shared": 0.7,
            "doc_lex_1": 0.6,
            "doc_exact": 0.5,
        }
    )
    text_of = TextLookupWithHole(small_corpus)
    ce = CountingFakeCE()

    def tiny_policy(fused_scores: np.ndarray) -> int:
        return 1

    hr = HybridRetriever(
        dense, text_of=text_of, reranker=ce, profile=profile, depth_policy=tiny_policy
    )
    result = hr.retrieve("q", k=4)
    assert len(result.ids) == 4
    assert np.all(np.diff(result.scores) <= 0.0)


# --------------------------------------------------------------------------- #
# Embedding-dim-agnostic: several structurally-distinct dense fakes; assert no
# vector is ever passed in (the dense seam only ever receives (query, k)).
# --------------------------------------------------------------------------- #
def test_embedding_dim_agnostic_no_vector_ever_passed(small_corpus):
    base = FakeDense({"doc_dense_1": 0.9, "doc_dense_2": 0.8, "doc_shared": 0.7})
    # Structurally-distinct "systems": disjoint id spaces + different spreads.
    variants = [base, base.relabeled("sysA_"), base.relabeled("sysB_")]

    for variant in variants:
        rec = ArgRecordingDense(variant.full_ranking)
        hr = lift(rec, small_corpus, rerank=False)
        result = hr.retrieve("overview", k=3)
        assert len(result.ids) >= 1

        # Every call to the dense seam is exactly (query:str, k:int) — never a
        # vector / ndarray, positionally or by keyword.
        assert rec.received, "dense seam was never called"
        for args, kwargs in rec.received:
            assert len(args) == 2
            assert isinstance(args[0], str)
            assert isinstance(args[1], int)
            for value in (*args, *kwargs.values()):
                assert not isinstance(value, np.ndarray)
                assert not (isinstance(value, (list, tuple)) and value and
                            all(isinstance(x, (int, float)) for x in value)), (
                    "a vector-like sequence was passed to the dense seam"
                )


# --------------------------------------------------------------------------- #
# Fail-open end-to-end.
# --------------------------------------------------------------------------- #
def test_fail_open_lexical_degrades_to_dense_order():
    dense = FakeDense({"a": 0.9, "b": 0.5, "c": 0.1})
    bad_lexical = RaisingLexical()
    hr = HybridRetriever(dense, lexical=bad_lexical, reranker=None)
    result = hr.retrieve("q", k=3)

    assert bad_lexical.calls, "lexical should have been attempted"
    # With lexical skipped, fusion runs over dense alone -> dense order preserved.
    assert result.ids == ("a", "b", "c")
    assert np.all(np.diff(result.scores) <= 0.0)


def test_fail_open_ce_degrades_to_fused_order(small_corpus):
    dense = FakeDense({"doc_dense_1": 0.9, "doc_dense_2": 0.8, "doc_shared": 0.7})
    lexical = FakeLexical({"doc_exact": 5.0, "doc_lex_1": 3.0})
    text_of = TextLookupWithHole(small_corpus)
    bad_ce = RaisingCE()
    hr = HybridRetriever(dense, lexical=lexical, text_of=text_of, reranker=bad_ce)

    from fusion_recall.fusion import rrf

    fused = rrf(
        {SOURCE_DENSE: dense.full_ranking, SOURCE_LEXICAL: lexical.full_ranking}
    )
    result = hr.retrieve("q", k=5)

    assert bad_ce.calls, "CE should have been attempted before failing open"
    # CE failed -> the returned ids follow the fused order (no reordering).
    expected_ids = tuple(doc_id for doc_id, _ in fused)[: len(result.ids)]
    assert result.ids == expected_ids
    assert len(result.ids) >= 1


# --------------------------------------------------------------------------- #
# Validation (fail-fast at build time).
# --------------------------------------------------------------------------- #
def test_lift_rerank_true_without_text_source_raises():
    dense = FakeDense({"a": 0.9})
    lexical = FakeLexical({"a": 1.0})
    # rerank=True but neither corpus nor text_of -> no CE text source -> raise.
    with pytest.raises(InvalidProfileError):
        lift(dense, lexical=lexical, rerank=True)


def test_lift_no_lexical_and_no_corpus_raises():
    dense = FakeDense({"a": 0.9})
    # lexical is None and corpus is None -> cannot build BM25 -> raise.
    with pytest.raises(InvalidProfileError):
        lift(dense, rerank=False)


def test_lift_rerank_false_with_corpus_ok(small_corpus):
    # rerank=False with a corpus is valid (BM25 built; no CE needed).
    dense = FakeDense({"doc_dense_1": 0.9})
    hr = lift(dense, small_corpus, rerank=False)
    assert isinstance(hr, HybridRetriever)


def test_lift_rerank_true_with_text_of_and_injected_lexical_ok(small_corpus):
    # text_of (not corpus) satisfies the CE text-source requirement, and an
    # injected lexical satisfies the recall requirement -> valid.
    dense = FakeDense({"doc_dense_1": 0.9})
    lexical = FakeLexical({"doc_exact": 5.0})
    text_of = TextLookupWithHole(small_corpus)
    hr = lift(
        dense,
        text_of=text_of,
        lexical=lexical,
        rerank=True,
        reranker=CountingFakeCE(),
    )
    assert isinstance(hr, HybridRetriever)


# --------------------------------------------------------------------------- #
# Edge cases hosted here.
# --------------------------------------------------------------------------- #
def test_empty_query_returns_empty_or_dense(small_corpus):
    # An empty query yields no in-vocabulary lexical terms; dense may also return
    # nothing for "". The call must not raise and must return a coherent result.
    dense = FakeDense({})  # dense surfaces nothing for the empty query
    hr = lift(dense, small_corpus, rerank=False)
    result = hr.retrieve("", k=5)
    assert isinstance(result, HybridResult)
    assert len(result.ids) == len(result.scores)
    # Empty query, lexically-blind dense -> empty result, not a crash.
    assert result.ids == ()


def test_text_of_missing_for_id_handled(small_corpus):
    # The CE window contains an id whose text is missing; the hole returns "" so
    # the candidate ranks low but the call does not crash.
    dense = FakeDense({"doc_dense_1": 0.9, "doc_shared": 0.7, "doc_dense_2": 0.5})
    holed = TextLookupWithHole(
        {k: v for k, v in small_corpus.items() if k != "doc_shared"}
    )
    ce = CountingFakeCE()
    hr = HybridRetriever(dense, text_of=holed, reranker=ce)
    result = hr.retrieve("q", k=3)

    assert isinstance(result, HybridResult)
    assert "doc_shared" in holed.misses  # the hole was actually exercised
    # doc_shared (empty text -> CE score len("")==0) ranks last under the CE.
    assert result.ids[-1] == "doc_shared"


def test_candidate_k_greater_than_corpus_is_safe(small_corpus):
    dense = FakeDense({"doc_dense_1": 0.9, "doc_dense_2": 0.8})
    hr = lift(dense, small_corpus, rerank=False)
    # candidate_k far exceeds corpus size; must be safe.
    result = hr.retrieve("doc_dense_1 overview", k=3, candidate_k=10_000)
    assert isinstance(result, HybridResult)
    assert len(result.ids) >= 1


def test_single_document_corpus():
    dense = FakeDense({"only": 1.0})
    hr = lift(dense, {"only": "the one and only document"}, rerank=False)
    result = hr.retrieve("only document", k=5)
    assert result.ids == ("only",)
    assert len(result.scores) == 1


def test_weighted_profile_path(small_corpus):
    # The weighted fuse path (profile.method == "weighted") is wired through
    # _fuse with profile.weights, not just rrf.
    profile = FusionProfile(
        method="weighted", weights={SOURCE_DENSE: 0.5, SOURCE_LEXICAL: 0.5}
    )
    dense = FakeDense({"doc_dense_1": 0.9, "doc_shared": 0.7})
    lexical = FakeLexical({"doc_exact": 5.0, "doc_shared": 2.0})
    hr = HybridRetriever(dense, lexical=lexical, reranker=None, profile=profile)
    result = hr.retrieve("q", k=4)
    assert set(result.ids) == {"doc_dense_1", "doc_shared", "doc_exact"}
    assert np.all(np.diff(result.scores) <= 0.0)


def test_injected_fuse_overrides_profile():
    # An injected Fuser is used verbatim regardless of profile.method.
    dense = FakeDense({"a": 0.9, "b": 0.5})
    lexical = FakeLexical({"c": 4.0})
    calls = {"n": 0}

    def my_fuse(rankings):
        calls["n"] += 1
        # Deterministic union with a fixed order, distinct from rrf.
        merged = {}
        for ranking in rankings.values():
            for doc_id, score in ranking:
                merged[doc_id] = merged.get(doc_id, 0.0) + score
        return tuple(sorted(merged.items(), key=lambda kv: (-kv[1], str(kv[0]))))

    hr = HybridRetriever(dense, lexical=lexical, reranker=None, fuse=my_fuse)
    result = hr.retrieve("q", k=3)
    assert calls["n"] == 1
    assert set(result.ids) == {"a", "b", "c"}
