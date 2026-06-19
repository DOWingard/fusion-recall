"""Structural contract tests.

These tests pin the public surface of ``fusion_recall.contracts``: the seven
runtime-checkable Protocols, the two frozen dataclasses (with the
removal of ``ce_blend``), the exception hierarchy, and the module constants.
They are written before ``contracts.py`` exists and must fail (ImportError)
until it does.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from fusion_recall.contracts import (
    DEFAULT_CE_MODEL,
    DEFAULT_PROFILE,
    SOURCE_DENSE,
    SOURCE_LEXICAL,
    CrossEncoder,
    DenseRetriever,
    EmptyCorpusError,
    Fuser,
    FusionError,
    FusionProfile,
    HybridResult,
    InvalidProfileError,
    LexicalRetriever,
    Normalize,
    TextLookup,
    Tokenizer,
)


# --------------------------------------------------------------------------- #
# Conforming / non-conforming stubs for the runtime_checkable Protocols.
# Protocols carry a __call__ or a named method; conformance is structural.
# --------------------------------------------------------------------------- #


class _Callable:
    def __call__(self, *args, **kwargs):  # generic conforming __call__
        return ()


class _NotCallable:
    """No ``__call__`` and no Protocol methods -> conforms to nothing here."""


class _Scorer:
    def score(self, query, docs):
        return np.zeros(len(docs))


# --------------------------------------------------------------------------- #
# 1. Each Protocol is runtime_checkable: a conforming stub passes isinstance,
#    a non-conforming stub fails it.
# --------------------------------------------------------------------------- #

CALL_PROTOCOLS = [
    DenseRetriever,
    LexicalRetriever,
    Fuser,
    TextLookup,
    Tokenizer,
    Normalize,
]


@pytest.mark.parametrize("proto", CALL_PROTOCOLS)
def test_call_protocol_conforming_stub_passes(proto):
    assert isinstance(_Callable(), proto)


@pytest.mark.parametrize("proto", CALL_PROTOCOLS)
def test_call_protocol_nonconforming_stub_fails(proto):
    assert not isinstance(_NotCallable(), proto)


def test_cross_encoder_protocol_conforming_stub_passes():
    assert isinstance(_Scorer(), CrossEncoder)


def test_cross_encoder_protocol_nonconforming_stub_fails():
    # An object with __call__ but no .score must NOT satisfy CrossEncoder.
    assert not isinstance(_Callable(), CrossEncoder)


@pytest.mark.parametrize(
    "proto",
    [
        DenseRetriever,
        LexicalRetriever,
        CrossEncoder,
        Fuser,
        TextLookup,
        Tokenizer,
        Normalize,
    ],
)
def test_protocols_are_runtime_checkable(proto):
    # runtime_checkable Protocols accept isinstance() without raising TypeError.
    # (Non-runtime-checkable Protocols raise on isinstance.)
    try:
        isinstance(object(), proto)
    except TypeError:  # pragma: no cover - guards the @runtime_checkable decorator
        pytest.fail(f"{proto.__name__} is not @runtime_checkable")


# --------------------------------------------------------------------------- #
# 2. FusionProfile: frozen, correct defaults, NO ce_blend field.
# --------------------------------------------------------------------------- #


def test_fusion_profile_is_frozen_dataclass():
    assert dataclasses.is_dataclass(FusionProfile)
    p = FusionProfile()
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.method = "weighted"  # type: ignore[misc]


def test_fusion_profile_has_no_ce_blend():
    # Design-review removal: ce_blend must not exist anywhere on the profile.
    assert not hasattr(FusionProfile(), "ce_blend")
    field_names = {f.name for f in dataclasses.fields(FusionProfile)}
    assert "ce_blend" not in field_names


def test_fusion_profile_field_set_is_exact():
    field_names = {f.name for f in dataclasses.fields(FusionProfile)}
    assert field_names == {
        "method",
        "rrf_k",
        "weights",
        "rerank_top_m",
        "ce_sigmoid",
    }


def test_fusion_profile_defaults():
    p = FusionProfile()
    assert p.method == "rrf"
    assert p.rrf_k == 60
    assert p.rerank_top_m == 50
    assert p.ce_sigmoid is False


def test_fusion_profile_default_weights_keyed_by_source_constants():
    p = FusionProfile()
    assert dict(p.weights) == {SOURCE_DENSE: 0.5, SOURCE_LEXICAL: 0.5}
    assert set(p.weights.keys()) == {SOURCE_DENSE, SOURCE_LEXICAL}


def test_fusion_profile_default_weights_are_immutable():
    # A frozen dataclass must not expose a mutable shared default.
    p = FusionProfile()
    with pytest.raises(TypeError):
        p.weights["dense"] = 0.9  # type: ignore[index]


def test_fusion_profile_default_weights_not_shared_mutable_instance():
    # Two default-constructed profiles must not alias one mutable object that
    # could be mutated to corrupt the other.
    a = FusionProfile()
    b = FusionProfile()
    assert dict(a.weights) == dict(b.weights)


# --------------------------------------------------------------------------- #
# 3. HybridResult: frozen dataclass with the three named fields.
# --------------------------------------------------------------------------- #


def test_hybrid_result_is_frozen_dataclass():
    assert dataclasses.is_dataclass(HybridResult)
    r = HybridResult(
        ids=("a",),
        scores=np.array([1.0], dtype=np.float64),
        provenance={SOURCE_DENSE: np.array([1.0])},
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.ids = ("b",)  # type: ignore[misc]


def test_hybrid_result_field_set_is_exact():
    field_names = {f.name for f in dataclasses.fields(HybridResult)}
    assert field_names == {"ids", "scores", "provenance"}


def test_hybrid_result_round_trips_fields():
    ids = ("a", "b")
    scores = np.array([2.0, 1.0], dtype=np.float64)
    prov = {"fused": np.array([2.0, 1.0])}
    r = HybridResult(ids=ids, scores=scores, provenance=prov)
    assert r.ids == ids
    assert r.scores is scores
    assert r.provenance is prov


# --------------------------------------------------------------------------- #
# 4. Exception hierarchy: EmptyCorpusError / InvalidProfileError -> FusionError
#    -> ValueError.
# --------------------------------------------------------------------------- #


def test_fusion_error_subclasses_value_error():
    assert issubclass(FusionError, ValueError)


def test_empty_corpus_error_subclasses_fusion_error():
    assert issubclass(EmptyCorpusError, FusionError)
    assert issubclass(EmptyCorpusError, ValueError)


def test_invalid_profile_error_subclasses_fusion_error():
    assert issubclass(InvalidProfileError, FusionError)
    assert issubclass(InvalidProfileError, ValueError)


def test_subclass_exceptions_are_catchable_as_fusion_error():
    with pytest.raises(FusionError):
        raise EmptyCorpusError("x")
    with pytest.raises(FusionError):
        raise InvalidProfileError("y")


# --------------------------------------------------------------------------- #
# 5. Module constants.
# --------------------------------------------------------------------------- #


def test_default_ce_model_string():
    assert DEFAULT_CE_MODEL == "cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_source_constants():
    assert SOURCE_DENSE == "dense"
    assert SOURCE_LEXICAL == "lexical"


def test_default_profile_is_a_fusion_profile_with_spec_values():
    assert isinstance(DEFAULT_PROFILE, FusionProfile)
    assert DEFAULT_PROFILE.method == "rrf"
    assert DEFAULT_PROFILE.rrf_k == 60
    assert DEFAULT_PROFILE.rerank_top_m == 50
    assert DEFAULT_PROFILE.ce_sigmoid is False


def test_default_profile_weights_keyed_by_exactly_the_two_constants():
    assert set(DEFAULT_PROFILE.weights.keys()) == {SOURCE_DENSE, SOURCE_LEXICAL}


# --------------------------------------------------------------------------- #
# 6. Ranking / ID aliases are importable and shaped as documented.
# --------------------------------------------------------------------------- #


def test_ranking_and_id_aliases_exist():
    from fusion_recall.contracts import ID, Ranking  # noqa: F401

    # Aliases are usable for annotation / construction; a concrete Ranking value
    # is a tuple of (id, float) pairs.
    sample: Ranking = (("a", 1.0), ("b", 0.5))
    assert sample[0][0] == "a"
    assert sample[0][1] == 1.0


# --------------------------------------------------------------------------- #
# 7. The shared test doubles (tests/conftest.py) genuinely satisfy the seam
#    Protocols. Chunks 2-5 rely on isinstance(double, Protocol) being True, so
#    that contract is pinned here.
# --------------------------------------------------------------------------- #


def test_fake_dense_satisfies_dense_retriever():
    from tests.conftest import FakeDense

    d = FakeDense({"a": 1.0, "b": 0.5})
    assert isinstance(d, DenseRetriever)


def test_fake_lexical_satisfies_lexical_retriever():
    from tests.conftest import FakeLexical

    lex = FakeLexical({"a": 1.0})
    assert isinstance(lex, LexicalRetriever)


def test_raising_lexical_satisfies_lexical_retriever():
    from tests.conftest import RaisingLexical

    assert isinstance(RaisingLexical(), LexicalRetriever)


def test_counting_fake_ce_satisfies_cross_encoder():
    from tests.conftest import CountingFakeCE

    assert isinstance(CountingFakeCE(), CrossEncoder)


def test_raising_ce_satisfies_cross_encoder():
    from tests.conftest import RaisingCE

    assert isinstance(RaisingCE(), CrossEncoder)


def test_text_lookup_with_hole_satisfies_text_lookup():
    from tests.conftest import TextLookupWithHole

    assert isinstance(TextLookupWithHole({"a": "text"}), TextLookup)


def test_fake_dense_records_query_and_k_and_truncates():
    from tests.conftest import FakeDense

    d = FakeDense({"a": 1.0, "b": 0.5, "c": 0.25})
    out = d("q", 2)
    assert out == (("a", 1.0), ("b", 0.5))  # descending, truncated to k
    assert d.calls == [("q", 2)]


def test_fake_dense_without_omits_doc_lexical_blind():
    from tests.conftest import FakeDense

    d = FakeDense({"a": 1.0, "b": 0.5, "exact": 0.9})
    blind = d.without("exact")
    ids = [doc_id for doc_id, _ in blind("q", 10)]
    assert "exact" not in ids
    assert "a" in ids and "b" in ids


def test_fake_dense_relabeled_is_structurally_distinct():
    from tests.conftest import FakeDense

    d = FakeDense({"a": 1.0, "b": 0.5})
    other = d.relabeled("sysB_")
    ids = {doc_id for doc_id, _ in other("q", 10)}
    # Disjoint id space from the original -> a distinct "system".
    assert ids == {"sysB_a", "sysB_b"}


def test_counting_fake_ce_counts_calls_and_doc_lengths():
    from tests.conftest import CountingFakeCE

    ce = CountingFakeCE()
    scores = ce.score("q", ["aa", "bbbb"])
    assert scores.dtype == np.float64
    assert scores.shape == (2,)
    # Default rule: score == len(text); deterministic, general (not hardcoded).
    assert list(scores) == [2.0, 4.0]
    assert ce.calls == 1
    assert ce.doc_counts == [2]


def test_counting_fake_ce_honors_score_table():
    from tests.conftest import CountingFakeCE

    ce = CountingFakeCE(scores_by_text={"hit": 99.0})
    scores = ce.score("q", ["hit", "miss"])
    assert scores[0] == 99.0
    assert scores[1] == float(len("miss"))


def test_raising_doubles_raise_on_call():
    from tests.conftest import RaisingCE, RaisingLexical

    lex = RaisingLexical()
    with pytest.raises(RuntimeError):
        lex("q", 5)
    assert lex.calls == [("q", 5)]

    ce = RaisingCE()
    with pytest.raises(RuntimeError):
        ce.score("q", ["d"])
    assert ce.calls == 1


def test_text_lookup_with_hole_returns_empty_for_missing_and_records():
    from tests.conftest import TextLookupWithHole

    lookup = TextLookupWithHole({"a": "alpha"})
    assert lookup("a") == "alpha"
    assert lookup("missing") == ""  # chosen behavior: empty string, no crash
    assert lookup.misses == ["missing"]
