"""Tests for the BM25 lexical index.

These are the contracts plus the edge cases that belong to
BM25 (single-document corpus; ``k`` greater than corpus size returns all
available). Written test-first per the TDD protocol.

The reference Okapi BM25 expected values below are derived **by hand from the
formula**, not copied from the implementation's output. The full arithmetic is
shown in :data:`REFERENCE_DERIVATION` and re-checked inline at each assertion, so
the implementation has to compute the score generally to pass — it cannot be
satisfied by a hardcoded table.

Formula under test (non-negative IDF variant)::

    idf(t)   = ln(1 + (N - df + 0.5) / (df + 0.5))
    score(d) = Σ_t  idf(t) * tf(t,d) * (k1 + 1)
                          / (tf(t,d) + k1 * (1 - b + b * dl(d) / avgdl))

with k1 = 1.5, b = 0.75.
"""

from __future__ import annotations

import math

import pytest

from fusion_recall.bm25 import BM25Index, default_tokenizer
from fusion_recall.contracts import EmptyCorpusError, LexicalRetriever

# --------------------------------------------------------------------------- #
# Hand-derived reference corpus and BM25 values (the anti-cheat crux).
#
# Corpus (default tokenizer = lowercase + split on non-alphanumeric):
#   doc1: "the cat sat"       -> [the, cat, sat]        dl = 3
#   doc2: "the dog ran"       -> [the, dog, ran]        dl = 3
#   doc3: "the cat ran fast"  -> [the, cat, ran, fast]  dl = 4
#
#   N = 3 documents
#   total tokens = 3 + 3 + 4 = 10  ->  avgdl = 10 / 3 = 3.33333...
#   df: the=3, cat=2, ran=2, sat=1, dog=1, fast=1
#   k1 = 1.5, b = 0.75
#
# IDF (idf = ln(1 + (N - df + 0.5)/(df + 0.5))):
#   idf(cat) = idf(ran) = ln(1 + (3 - 2 + 0.5)/(2 + 0.5))
#                       = ln(1 + 1.5/2.5) = ln(1.6)        = 0.47000362924...
#   idf(the)            = ln(1 + (3 - 3 + 0.5)/(3 + 0.5))
#                       = ln(1 + 0.5/3.5) = ln(8/7)        = 0.13353139262...
#
# tf-normalization denominator factor  k1*(1 - b + b*dl/avgdl):
#   dl=3: 1.5*(0.25 + 0.75 * 3/(10/3)) = 1.5*(0.25 + 0.675)        = 1.3875
#   dl=4: 1.5*(0.25 + 0.75 * 4/(10/3)) = 1.5*(0.25 + 0.9)          = 1.725
#   tfnorm(tf=1, dl=3) = 1*(1.5+1)/(1 + 1.3875) = 2.5/2.3875 = 1.04712041884...
#   tfnorm(tf=1, dl=4) = 1*(1.5+1)/(1 + 1.725 ) = 2.5/2.725  = 0.91743119266...
#
# Query "cat ran"  (query terms: cat, ran):
#   doc1 "the cat sat"      -> cat tf=1, ran tf=0  (dl=3)
#       score = idf(cat)*tfnorm(1,3)
#             = 0.47000362924 * 1.04712041884 = 0.49215039711...
#   doc2 "the dog ran"      -> cat tf=0, ran tf=1  (dl=3)
#       score = idf(ran)*tfnorm(1,3)
#             = 0.47000362924 * 1.04712041884 = 0.49215039711...   (== doc1)
#   doc3 "the cat ran fast" -> cat tf=1, ran tf=1  (dl=4)
#       score = (idf(cat)+idf(ran))*tfnorm(1,4)
#             = 2 * 0.47000362924 * 0.91743119266 = 0.86239198026...
#
# Expected ranking for "cat ran":  doc3 > {doc1, doc2}.  doc1 and doc2 tie
# exactly; the index breaks the tie deterministically by original id order, so
# doc1 precedes doc2.
# --------------------------------------------------------------------------- #
REFERENCE_DERIVATION = "see module docstring / the comment block above"

_IDS = ("doc1", "doc2", "doc3")
_TEXTS = ("the cat sat", "the dog ran", "the cat ran fast")

# Hand-computed expected scores for query "cat ran" (NOT taken from the impl).
_EXP_DOC1 = 0.47000362924573563 * 1.0471204188481675  # 0.49215039711...
_EXP_DOC2 = _EXP_DOC1  # identical structure to doc1
_EXP_DOC3 = 2 * 0.47000362924573563 * 0.9174311926605506  # 0.86239198026...


@pytest.fixture
def index() -> BM25Index:
    return BM25Index.fit(_IDS, _TEXTS)


# --------------------------------------------------------------------------- #
# Reference Okapi BM25 scores on a tiny hand-computed corpus.
# --------------------------------------------------------------------------- #


def test_scores_match_hand_computed_reference(index: BM25Index) -> None:
    """Returned scores equal the by-hand Okapi BM25 values (k1=1.5, b=0.75)."""
    ranking = index("cat ran", k=3)
    scores = {doc_id: score for doc_id, score in ranking}

    # Re-derive each expected value from first principles right here so the
    # assertion is anchored to the formula, not to any stored constant.
    avgdl = 10 / 3
    idf_cat = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))

    def tfnorm(tf: int, dl: int) -> float:
        return tf * (1.5 + 1) / (tf + 1.5 * (1 - 0.75 + 0.75 * dl / avgdl))

    exp_doc1 = idf_cat * tfnorm(1, 3)
    exp_doc3 = 2 * idf_cat * tfnorm(1, 4)

    assert scores["doc1"] == pytest.approx(exp_doc1)
    assert scores["doc2"] == pytest.approx(exp_doc1)
    assert scores["doc3"] == pytest.approx(exp_doc3)
    # Cross-check against the independently stored literals.
    assert scores["doc1"] == pytest.approx(_EXP_DOC1)
    assert scores["doc3"] == pytest.approx(_EXP_DOC3)


def test_single_term_query_scores(index: BM25Index) -> None:
    """A single-term query scores exactly idf(term) * tfnorm(tf, dl) per doc."""
    ranking = index("cat", k=3)
    scores = {doc_id: score for doc_id, score in ranking}

    avgdl = 10 / 3
    idf_cat = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))

    def tfnorm(tf: int, dl: int) -> float:
        return tf * (1.5 + 1) / (tf + 1.5 * (1 - 0.75 + 0.75 * dl / avgdl))

    # "cat" occurs in doc1 (dl=3) and doc3 (dl=4); not in doc2.
    assert scores["doc1"] == pytest.approx(idf_cat * tfnorm(1, 3))
    assert scores["doc3"] == pytest.approx(idf_cat * tfnorm(1, 4))
    assert "doc2" not in scores  # zero-scoring docs are not returned


# --------------------------------------------------------------------------- #
# IDF uses GLOBAL document frequencies; ubiquitous term contributes ≈0.
# --------------------------------------------------------------------------- #


def test_idf_uses_global_df_ubiquitous_term_near_zero() -> None:
    """A term in every document has IDF ~ 0 and far below a rare term's IDF.

    Built on a larger corpus (N=20) where ``common`` appears in every document
    and ``rareterm`` in exactly one. With the non-negative IDF variant the
    ubiquitous term's IDF -> 0 as N grows; here it must be tiny and strictly
    dominated by the rare term.
    """
    n = 20
    ids = [f"d{i}" for i in range(n)]
    texts = [f"common filler{i}" for i in range(n)]
    texts[0] = "common rareterm"  # rareterm appears in exactly one document
    idx = BM25Index.fit(ids, texts)

    common_only = {d: s for d, s in idx("common", k=n)}
    # Every doc contains "common", so the score is its (small) idf scaled by the
    # tf-normalization. The idf itself is ln(1 + 0.5/20.5) ~ 0.0241.
    idf_common = math.log(1 + (n - n + 0.5) / (n + 0.5))
    assert idf_common < 0.05  # "contributes ~0"

    # The rare term's idf is much larger than the ubiquitous term's.
    idf_rare = math.log(1 + (n - 1 + 0.5) / (1 + 0.5))
    assert idf_rare > idf_common * 10

    # And in retrieval, doc0's "rareterm" score dwarfs any "common" score.
    rare_only = {d: s for d, s in idx("rareterm", k=n)}
    assert set(rare_only) == {"d0"}
    assert rare_only["d0"] > max(common_only.values())


# --------------------------------------------------------------------------- #
# Ranking orders documents correctly for a known query.
# --------------------------------------------------------------------------- #


def test_ranking_order_for_known_query(index: BM25Index) -> None:
    """doc3 (both query terms) ranks above doc1/doc2 (one term each)."""
    ranking = index("cat ran", k=3)
    ids = [doc_id for doc_id, _ in ranking]
    assert ids[0] == "doc3"
    assert set(ids[1:]) == {"doc1", "doc2"}


def test_ranking_is_descending(index: BM25Index) -> None:
    """The returned Ranking is sorted strictly non-increasing by score."""
    ranking = index("cat ran", k=3)
    scores = [score for _, score in ranking]
    assert scores == sorted(scores, reverse=True)


# --------------------------------------------------------------------------- #
# Add / remove mutate results correctly.
# --------------------------------------------------------------------------- #


def test_remove_excludes_document(index: BM25Index) -> None:
    """remove(x) then query no longer returns x."""
    before = {d for d, _ in index("cat ran", k=3)}
    assert "doc3" in before
    index.remove("doc3")
    after = {d for d, _ in index("cat ran", k=3)}
    assert "doc3" not in after
    assert after == {"doc1", "doc2"}


def test_add_includes_document(index: BM25Index) -> None:
    """add(x) then query can return x."""
    before = {d for d, _ in index("fast", k=3)}
    assert before == {"doc3"}
    index.add("doc4", "fast fast lane")
    after = {d for d, _ in index("fast", k=3)}
    assert "doc4" in after


def test_remove_then_add_round_trip_changes_stats(index: BM25Index) -> None:
    """Removing then re-adding a doc updates global stats (df / avgdl)."""
    base = {d: s for d, s in index("cat ran", k=3)}
    index.remove("doc2")  # "the dog ran" -> ran df drops 2->1, avgdl changes
    # ran is now rarer -> doc3's score (contains ran) should shift; the result
    # set must drop doc2 and still contain doc1 and doc3.
    after = {d for d, _ in index("cat ran", k=3)}
    assert after == {"doc1", "doc3"}
    # doc1 only has "cat" (df unchanged at 2) but avgdl changed (10->7, /2),
    # so its score must differ from the original three-doc score.
    after_scores = {d: s for d, s in index("cat ran", k=3)}
    assert after_scores["doc1"] != pytest.approx(base["doc1"])


def test_remove_unknown_id_raises(index: BM25Index) -> None:
    """Removing an id that is not in the index is an error, not a silent no-op."""
    with pytest.raises(KeyError):
        index.remove("nonexistent")


def test_len_tracks_corpus_size(index: BM25Index) -> None:
    """__len__ reflects add/remove."""
    assert len(index) == 3
    index.add("doc4", "another doc")
    assert len(index) == 4
    index.remove("doc1")
    assert len(index) == 3


# --------------------------------------------------------------------------- #
# Tokenizer: default lowercases + splits on non-alphanumeric; custom honored.
# --------------------------------------------------------------------------- #


def test_default_tokenizer_lowercases_and_splits_non_alphanumeric() -> None:
    assert default_tokenizer("Hello, World!") == ["hello", "world"]
    # ``\w`` keeps the underscore (identifier-preserving) but splits on the
    # hyphen and other punctuation/whitespace.
    assert default_tokenizer("CamelCase_snake-kebab") == ["camelcase_snake", "kebab"]
    assert default_tokenizer("error_code 42  ABC") == ["error_code", "42", "abc"]
    assert default_tokenizer("a.b,c;d") == ["a", "b", "c", "d"]
    assert default_tokenizer("   ") == []
    assert default_tokenizer("") == []


def test_default_tokenizer_keeps_alphanumeric_runs() -> None:
    # Underscore is alphanumeric-ish in identifiers? No: \w includes underscore.
    # default_tokenizer splits on NON-alphanumeric; \w (which includes "_") is
    # the kept run, so "error_code" stays one token.
    assert default_tokenizer("v1.2.3-rc4") == ["v1", "2", "3", "rc4"]


def test_custom_tokenizer_is_honored() -> None:
    """A custom tokenizer changes which terms match."""

    def whole_text_tokenizer(text: str) -> list[str]:
        # Treat the entire (stripped, lowercased) string as a single token.
        stripped = text.strip().lower()
        return [stripped] if stripped else []

    idx = BM25Index.fit(
        ["a", "b"],
        ["alpha beta", "alpha beta"],
        tokenizer=whole_text_tokenizer,
    )
    # Under the default tokenizer "alpha" would match; under this one only the
    # exact full string is a term.
    assert idx("alpha", k=2) == ()
    got = {d for d, _ in idx("alpha beta", k=2)}
    assert got == {"a", "b"}


# --------------------------------------------------------------------------- #
# Empty corpus raises; OOV query returns empty Ranking.
# --------------------------------------------------------------------------- #


def test_fit_empty_corpus_raises() -> None:
    with pytest.raises(EmptyCorpusError):
        BM25Index.fit([], [])


def test_fit_mismatched_lengths_raises() -> None:
    """ids and texts of different lengths is a construction error."""
    with pytest.raises(ValueError):
        BM25Index.fit(["a", "b"], ["only one text"])


def test_out_of_vocabulary_query_returns_empty_ranking(index: BM25Index) -> None:
    """A query whose terms are absent from the corpus returns () (not an error)."""
    assert index("zzz nonexistentterm", k=3) == ()


def test_empty_query_returns_empty_ranking(index: BM25Index) -> None:
    assert index("", k=3) == ()
    assert index("   ", k=3) == ()


# --------------------------------------------------------------------------- #
# Protocol conformance + descending Ranking shape.
# --------------------------------------------------------------------------- #


def test_is_lexical_retriever(index: BM25Index) -> None:
    assert isinstance(index, LexicalRetriever)


def test_returns_ranking_tuple_shape(index: BM25Index) -> None:
    ranking = index("cat ran", k=3)
    assert isinstance(ranking, tuple)
    for entry in ranking:
        assert isinstance(entry, tuple) and len(entry) == 2
        _doc_id, score = entry
        assert isinstance(score, float)


# --------------------------------------------------------------------------- #
# Determinism: identical corpus + query -> identical Ranking.
# --------------------------------------------------------------------------- #


def test_determinism_same_inputs_same_ranking() -> None:
    a = BM25Index.fit(_IDS, _TEXTS)
    b = BM25Index.fit(_IDS, _TEXTS)
    assert a("cat ran", k=3) == b("cat ran", k=3)


def test_determinism_tie_break_is_stable(index: BM25Index) -> None:
    """Exact-tie docs (doc1, doc2 for 'cat ran') order by original id, stably."""
    r1 = index("cat ran", k=3)
    r2 = index("cat ran", k=3)
    assert r1 == r2
    ids = [d for d, _ in r1]
    # doc1 and doc2 tie exactly; deterministic tie-break puts doc1 before doc2.
    assert ids.index("doc1") < ids.index("doc2")


# --------------------------------------------------------------------------- #
# Edge cases owned by BM25: single-document corpus; k > corpus size.
# --------------------------------------------------------------------------- #


def test_single_document_corpus() -> None:
    """A one-document corpus is valid and queryable.

    With N=1, df=1 for every present term, so idf = ln(1 + 0.5/1.5) = ln(4/3),
    and dl == avgdl so the length-normalization factor is exactly k1.
    """
    idx = BM25Index.fit(["only"], ["alpha beta gamma"])
    assert len(idx) == 1
    ranking = idx("alpha", k=5)
    assert len(ranking) == 1
    (doc_id, score) = ranking[0]
    assert doc_id == "only"
    # dl == avgdl (== 3) so tfnorm = tf*(k1+1)/(tf + k1*(1-b+b)) = tf*(k1+1)/(tf+k1).
    idf = math.log(1 + (1 - 1 + 0.5) / (1 + 0.5))
    tfnorm = 1 * (1.5 + 1) / (1 + 1.5)
    assert score == pytest.approx(idf * tfnorm)


def test_k_greater_than_corpus_returns_all_available(index: BM25Index) -> None:
    """k larger than the number of scoring docs returns all of them, no error."""
    ranking = index("the", k=100)  # 'the' is in all 3 docs
    assert len(ranking) == 3
    assert {d for d, _ in ranking} == {"doc1", "doc2", "doc3"}


def test_k_truncates_to_top_k(index: BM25Index) -> None:
    """A k smaller than the candidate count returns exactly the top k."""
    ranking = index("the", k=1)  # all three docs score; return only the top
    assert len(ranking) == 1


def test_zero_scoring_docs_excluded() -> None:
    """Documents with no query-term overlap are not padded into the result."""
    idx = BM25Index.fit(["x", "y"], ["alpha", "beta"])
    ranking = idx("alpha", k=5)
    assert {d for d, _ in ranking} == {"x"}
