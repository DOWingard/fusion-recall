"""In-repo evaluation dataset and a deterministic, lexical-blind dense stand-in.

The corpus and queries are chosen so relevance is decided by **lexical** signal —
exact function/symbol names, error codes, rare tokens, UUID/version strings — the
class of relevance that dense embeddings smear into "nearby concept" space and
therefore miss. Each query's qrels mark the documents that contain
its decisive lexical signal.

The dense baseline is a **deterministic stand-in**, not a real embedding model.
Its job is *lever isolation*: holding the dense ranking fixed and reproducible is
what makes the BM25 lever (``rrf - dense``) and the cross-encoder lever
(``hybrid - rrf``) separately meaningful. The stand-in models the embedding
recall ceiling honestly — it ranks documents by topical/conceptual word overlap
(a deterministic proxy for semantic similarity) but is **blind to rare
exact-match tokens**, so a document whose relevance is purely lexical never
enters its ranking. It is *not* rigged to be globally terrible: it ranks the
topical documents sensibly; it simply cannot see lexical-only relevance, exactly
as a real embedder cannot.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from fusion_recall.bm25 import default_tokenizer
from fusion_recall.contracts import ID, Ranking


@dataclass(frozen=True)
class Query:
    """An evaluation query: its text and the set of relevant document ids."""

    text: str
    relevant: frozenset[ID] = field(default_factory=frozenset)


# --------------------------------------------------------------------------- #
# Corpus. A mix of:
#  * topical/conceptual documents a dense embedder would rank well, and
#  * lexical-only documents whose relevance hinges on a rare exact token (an
#    error code, a symbol name, a UUID, a version string) that a dense embedder
#    would smear away.
# --------------------------------------------------------------------------- #
_CORPUS: dict[ID, str] = {
    # Topical documents about search / ranking — dense-friendly, shared concept
    # vocabulary, no rare exact tokens.
    "topic_search": (
        "an overview of information retrieval and search systems for ranking "
        "documents by relevance to a user query"
    ),
    "topic_ranking": (
        "ranking models order candidate documents so the most relevant results "
        "appear first in the search results list"
    ),
    "topic_embeddings": (
        "dense vector embeddings map text into a semantic space where nearby "
        "vectors represent conceptually similar passages"
    ),
    "topic_database": (
        "a relational database stores records in tables and answers queries "
        "with a structured query language engine"
    ),
    "topic_network": (
        "computer networks move packets between hosts using addressing and "
        "routing across interconnected links"
    ),
    # Lexical-only documents: relevance is carried by a rare exact token that
    # appears in no other document. A dense embedder smears these into generic
    # "concept" space; only lexical recall surfaces them.
    "lex_error_code": (
        "the parser aborts and raises error code E_4471_PARSE_FAIL when the "
        "input stream ends inside an unterminated token"
    ),
    "lex_symbol": (
        "the function compute_qhull_simplex_normals recomputes facet normals "
        "after an incremental insertion into the hull"
    ),
    "lex_uuid": (
        "the migration is tracked under id 7f3a9c12-uuid-4bd1-9a02-deadbeef0042 "
        "and must run exactly once per deployment"
    ),
    "lex_version": (
        "the regression was introduced in release v2.31.0-rc7 and reverted "
        "before the stable cut shipped to users"
    ),
    "lex_flag": (
        "passing the --no-coalesce-spans flag disables span merging in the "
        "tracing exporter for high cardinality debugging"
    ),
}

# --------------------------------------------------------------------------- #
# Queries. Each is dominated by a rare exact token whose only carrier is a single
# lexical-only document. The qrels mark that carrier. A couple of queries also
# include a topical document so the dense stand-in is not uniformly empty.
# --------------------------------------------------------------------------- #
_QUERIES: list[Query] = [
    Query(
        text="what raises error code E_4471_PARSE_FAIL",
        relevant=frozenset({"lex_error_code"}),
    ),
    Query(
        text="where is compute_qhull_simplex_normals defined",
        relevant=frozenset({"lex_symbol"}),
    ),
    Query(
        text="migration with id 7f3a9c12-uuid-4bd1-9a02-deadbeef0042",
        relevant=frozenset({"lex_uuid"}),
    ),
    Query(
        text="regression in release v2.31.0-rc7",
        relevant=frozenset({"lex_version"}),
    ),
    Query(
        text="what does the --no-coalesce-spans flag do for span merging",
        relevant=frozenset({"lex_flag"}),
    ),
    Query(
        # A query mixing a topical need with a lexical-only carrier: the dense
        # stand-in can recover the topical doc but still misses the lexical one.
        text="ranking search results and error code E_4471_PARSE_FAIL handling",
        relevant=frozenset({"lex_error_code", "topic_ranking"}),
    ),
]


# --------------------------------------------------------------------------- #
# Dense stand-in.
# --------------------------------------------------------------------------- #
# The lexical-only documents — relevance carried by a rare exact token. The dense
# stand-in is blind to these by construction (a real embedder smears such tokens
# into generic concept space and would not rank these documents for their token).
DENSE_BLIND_IDS: frozenset[ID] = frozenset(
    {"lex_error_code", "lex_symbol", "lex_uuid", "lex_version", "lex_flag"}
)

# A small "concept vocabulary": common topical words a dense embedder reasons
# over. The stand-in scores documents by overlap with the query on these words
# only — it cannot see rare exact tokens. This is the deterministic proxy for
# semantic similarity that gives the stand-in a sensible (non-degenerate) ranking
# over the topical documents while remaining lexical-blind.
_CONCEPT_VOCAB: frozenset[str] = frozenset(
    {
        "search",
        "ranking",
        "rank",
        "results",
        "relevance",
        "relevant",
        "documents",
        "document",
        "query",
        "queries",
        "retrieval",
        "embeddings",
        "embedding",
        "semantic",
        "vector",
        "vectors",
        "database",
        "network",
        "networks",
        "handling",
        "merging",
    }
)


def dense_standin(query: str, k: int) -> Ranking:
    """A deterministic, lexical-blind ``DenseRetriever`` over the eval corpus.

    Scores each *non-blind* document by the number of distinct concept-vocabulary
    terms it shares with the query (a deterministic proxy for semantic overlap).
    Documents in :data:`DENSE_BLIND_IDS` are never scored — the stand-in cannot
    see their rare exact-match relevance, mirroring an embedder's recall ceiling.
    Documents with no concept overlap are omitted (score 0 contributes nothing).
    Ties are broken by id for bit-identical reproducibility.
    """
    q_terms = set(default_tokenizer(query)) & _CONCEPT_VOCAB

    scored: dict[ID, float] = {}
    for doc_id, text in _CORPUS.items():
        if doc_id in DENSE_BLIND_IDS:
            continue  # lexical-only doc: invisible to the dense stand-in
        doc_terms = set(default_tokenizer(text)) & _CONCEPT_VOCAB
        overlap = len(q_terms & doc_terms)
        if overlap > 0:
            scored[doc_id] = float(overlap)

    ranked = sorted(scored.items(), key=lambda kv: (-kv[1], str(kv[0])))
    return tuple((doc_id, score) for doc_id, score in ranked[:k])


def load_dataset() -> tuple[Mapping[ID, str], list[Query]]:
    """Return the (corpus, queries) pair for the evaluation harness.

    The corpus is returned as a fresh dict and the queries as a fresh list so
    callers cannot mutate the module-level fixtures.
    """
    return dict(_CORPUS), list(_QUERIES)
