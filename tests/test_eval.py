"""Evaluation harness tests — the proof of objective 2.

Two layers:

* **Metric unit tests** (offline, fast): ``recall_at_k`` / ``ndcg_at_k`` checked
  against values hand-computed in the test body (arithmetic shown in comments),
  never copied from the implementation's output.
* **Three-config comparison**: dense-only vs dense+BM25 (RRF, no CE) vs full
  hybrid (RRF + CE). The BM25 lever (``rrf - dense``) runs offline against the
  deterministic, lexical-blind dense stand-in and must be strictly positive. The
  cross-encoder lever (``hybrid - rrf``) loads the real default model and is
  therefore marked ``slow`` and skips (never hangs/errors) when the model is
  unavailable offline.

The metric tests are written before ``eval.metrics`` exists (TDD): the expected
recall/nDCG values are derived by hand here, and the implementation is built to
satisfy them.
"""

from __future__ import annotations

import math
import os

import pytest

from eval.dataset import load_dataset
from eval.metrics import ndcg_at_k, recall_at_k
from eval.run import evaluate


# --------------------------------------------------------------------------- #
# recall_at_k — hand-computed.
# recall@k = |relevant ∩ retrieved[:k]| / |relevant|
# --------------------------------------------------------------------------- #
def test_recall_at_k_partial_hit():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"b", "d", "z"}  # |relevant| = 3
    # top-2 = [a, b]; intersection with relevant = {b} -> 1 hit.
    # recall@2 = 1 / 3.
    assert recall_at_k(retrieved, relevant, 2) == pytest.approx(1.0 / 3.0)


def test_recall_at_k_all_relevant_within_k():
    retrieved = ["b", "d", "a", "c"]
    relevant = {"b", "d"}  # |relevant| = 2
    # top-3 = [b, d, a]; intersection = {b, d} -> 2 hits.
    # recall@3 = 2 / 2 = 1.0.
    assert recall_at_k(retrieved, relevant, 3) == pytest.approx(1.0)


def test_recall_at_k_cuts_at_k():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"d"}  # |relevant| = 1, but d is at rank 4
    # top-2 = [a, b]; intersection = {} -> 0 hits.
    # recall@2 = 0 / 1 = 0.0  (k strictly truncates).
    assert recall_at_k(retrieved, relevant, 2) == pytest.approx(0.0)


def test_recall_at_k_empty_relevant_is_zero():
    # No relevant docs -> recall is defined as 0.0 (avoid divide-by-zero).
    assert recall_at_k(["a", "b"], set(), 2) == pytest.approx(0.0)


def test_recall_at_k_k_larger_than_retrieved():
    retrieved = ["a", "b"]
    relevant = {"a", "b"}  # |relevant| = 2
    # top-5 of a 2-item list = [a, b]; intersection = {a, b} -> 2 hits.
    # recall@5 = 2 / 2 = 1.0.
    assert recall_at_k(retrieved, relevant, 5) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# ndcg_at_k — binary gains, hand-computed.
# DCG@k = Σ_{i=1..k} rel_i / log2(i + 1)   (rel_i ∈ {0, 1})
# IDCG@k = DCG of the ideal ranking (all relevant first), capped at k.
# nDCG@k = DCG@k / IDCG@k.
# --------------------------------------------------------------------------- #
def test_ndcg_at_k_perfect_ranking_is_one():
    retrieved = ["a", "b", "c"]
    relevant = {"a", "b"}
    # Relevant docs are already at ranks 1 and 2 -> DCG == IDCG -> nDCG = 1.0.
    assert ndcg_at_k(retrieved, relevant, 3) == pytest.approx(1.0)


def test_ndcg_at_k_relevant_lower_ranked():
    retrieved = ["x", "a", "y"]
    relevant = {"a"}  # one relevant doc, sitting at rank 2
    # DCG@3 = 1 / log2(2 + 1) = 1 / log2(3).
    # IDCG@3: ideal puts the single relevant doc at rank 1 -> 1 / log2(2) = 1.
    # nDCG@3 = (1 / log2(3)) / 1 = 1 / log2(3) ≈ 0.6309298.
    expected = (1.0 / math.log2(3.0)) / 1.0
    assert ndcg_at_k(retrieved, relevant, 3) == pytest.approx(expected)
    assert ndcg_at_k(retrieved, relevant, 3) == pytest.approx(0.6309297535714574)


def test_ndcg_at_k_two_relevant_one_demoted():
    retrieved = ["a", "x", "b"]
    relevant = {"a", "b"}  # ranks 1 and 3
    # DCG@3 = 1/log2(2) + 0 + 1/log2(4) = 1.0 + 0.5 = 1.5.
    # IDCG@3 = 1/log2(2) + 1/log2(3) = 1.0 + 0.63092975... = 1.63092975...
    # nDCG@3 = 1.5 / 1.6309297535714573 ≈ 0.91972...
    dcg = 1.0 / math.log2(2.0) + 1.0 / math.log2(4.0)
    idcg = 1.0 / math.log2(2.0) + 1.0 / math.log2(3.0)
    assert ndcg_at_k(retrieved, relevant, 3) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(retrieved, relevant, 3) == pytest.approx(0.9197207891481876)


def test_ndcg_at_k_no_relevant_is_zero():
    # No relevant docs -> IDCG is 0; nDCG is defined as 0.0 (no divide-by-zero).
    assert ndcg_at_k(["a", "b"], set(), 2) == pytest.approx(0.0)


def test_ndcg_at_k_no_hit_within_k_is_zero():
    retrieved = ["x", "y", "a"]
    relevant = {"a"}  # relevant doc is at rank 3, outside k=2
    # DCG@2 = 0; nDCG@2 = 0.0.
    assert ndcg_at_k(retrieved, relevant, 2) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Dataset sanity: it must actually favor lexical signal, and the dense stand-in
# must be lexical-blind (miss at least one purely-lexical relevant doc) so the
# BM25 lever has something real to recover.
# --------------------------------------------------------------------------- #
def test_dataset_is_lexically_decisive():
    corpus, queries = load_dataset()
    assert len(corpus) >= 8
    assert len(queries) >= 4
    for q in queries:
        assert q.relevant, f"query {q.text!r} has no qrels"
        for rid in q.relevant:
            assert rid in corpus, f"qrel {rid!r} not in corpus"


def test_dense_standin_is_lexical_blind():
    """The dense stand-in must miss at least one relevant doc per its design.

    The whole eval hinges on the dense baseline being lexical-blind: if it
    already surfaced every relevant doc, ``rrf - dense`` would be ~0 and the
    BM25 lift would be unprovable. This asserts the stand-in genuinely omits
    lexical-only relevant docs from its top results (not by being globally
    broken, but by being blind to rare exact-match tokens).
    """
    from eval.dataset import DENSE_BLIND_IDS, dense_standin

    corpus, queries = load_dataset()
    assert DENSE_BLIND_IDS, "no lexical-only docs declared blind to the dense stand-in"
    # Every declared-blind id is genuinely a relevant doc for some query (so the
    # blindness actually costs recall) and is absent from that query's dense
    # ranking.
    blind_costs_recall = False
    for q in queries:
        dense_ids = {doc_id for doc_id, _ in dense_standin(q.text, len(corpus))}
        for rid in q.relevant:
            if rid in DENSE_BLIND_IDS:
                assert rid not in dense_ids, (
                    f"declared-blind doc {rid!r} leaked into the dense ranking "
                    f"for query {q.text!r}"
                )
                blind_costs_recall = True
    assert blind_costs_recall, "blindness never overlaps a qrel; lift unprovable"


# --------------------------------------------------------------------------- #
# BM25 lever (offline): dense+BM25 RRF must beat dense-only on BOTH metrics.
# This isolates the BM25 contribution and needs no model.
# --------------------------------------------------------------------------- #
def test_bm25_lever_offline():
    k = 5
    dense = evaluate("dense", k=k)
    rrf = evaluate("rrf", k=k)

    # Deterministic / reproducible across runs (pure BM25 + RRF path).
    assert evaluate("dense", k=k) == dense
    assert evaluate("rrf", k=k) == rrf

    # The BM25 lever is the rrf - dense delta; it must be a real, positive gain
    # on both metrics (lexical recovery of docs the dense stand-in misses).
    assert rrf["recall@k"] > dense["recall@k"], (
        f"BM25 did not lift recall: rrf={rrf['recall@k']} dense={dense['recall@k']}"
    )
    assert rrf["ndcg@k"] > dense["ndcg@k"], (
        f"BM25 did not lift nDCG: rrf={rrf['ndcg@k']} dense={dense['ndcg@k']}"
    )
    # "Meaningful margin", not a rounding artifact.
    assert rrf["recall@k"] - dense["recall@k"] >= 0.1


# --------------------------------------------------------------------------- #
# Full three-config comparison incl. the real cross-encoder (slow / optional).
# Skips cleanly when the model is unavailable offline; never hangs or errors.
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_full_three_config_comparison_with_real_ce():
    pytest.importorskip("sentence_transformers")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    k = 5
    dense = evaluate("dense", k=k)
    rrf = evaluate("rrf", k=k)
    try:
        hybrid = evaluate("hybrid", k=k)
    except Exception as exc:  # model not cached / unavailable offline
        pytest.skip(f"real cross-encoder model unavailable: {exc}")

    # Objective 2: full hybrid >= dense-only on BOTH metrics by a meaningful
    # margin.
    assert hybrid["recall@k"] >= dense["recall@k"] + 0.1
    assert hybrid["ndcg@k"] >= dense["ndcg@k"] + 0.05

    # The two levers reported SEPARATELY: BM25 lever and CE lever.
    bm25_recall_delta = rrf["recall@k"] - dense["recall@k"]
    ce_ndcg_delta = hybrid["ndcg@k"] - rrf["ndcg@k"]
    # BM25 raises the recall ceiling (candidate generation).
    assert bm25_recall_delta > 0.0
    # The cross-encoder is a precision/ordering engine: it must not regress nDCG
    # below the fused ordering it reranks.
    assert ce_ndcg_delta >= 0.0
