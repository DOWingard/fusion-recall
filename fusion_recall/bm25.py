"""Built-in BM25 lexical index and the default tokenizer.

``BM25Index`` is the batteries-included :class:`~fusion_recall.contracts.LexicalRetriever`:
an in-memory inverted index scoring queries with Okapi BM25. It is the lexical
half of the parallel recall stage and runs over the full corpus so the recall
ceiling is the union of dense and lexical, not dense alone.

Scoring uses the non-negative IDF variant so a term that appears in every
document contributes approximately zero rather than a negative penalty, and the
score never goes negative::

    idf(t)   = ln(1 + (N - df + 0.5) / (df + 0.5))
    score(d) = Σ_{t in query}  idf(t) * tf(t,d) * (k1 + 1)
                    / (tf(t,d) + k1 * (1 - b + b * dl(d) / avgdl))

The pure path is deterministic: identical corpus and query yield a bit-identical
``Ranking``, with exact-score ties broken by insertion order so re-runs are
stable. The host owns synchronization — :meth:`add` / :meth:`remove` keep the
index in step with the host's store and update the global statistics (document
frequencies, document lengths, average length) incrementally.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

from fusion_recall.contracts import ID, EmptyCorpusError, Ranking, Tokenizer

# Split on runs of non-alphanumeric characters. ``\w`` is unicode-aware and
# includes the underscore, so identifiers like ``error_code`` stay one token
# while punctuation, whitespace, and symbols are treated as delimiters.
_NON_ALNUM = re.compile(r"\w+")


def default_tokenizer(text: str) -> list[str]:
    """Lowercase ``text`` and split it into alphanumeric (``\\w``) runs."""
    return _NON_ALNUM.findall(text.lower())


class BM25Index:
    """In-memory Okapi BM25 index implementing ``LexicalRetriever``.

    Build with :meth:`fit`; query by calling the instance. The index keeps an
    inverted index (term -> {doc id -> term frequency}), each document's length,
    and the running total length so the average document length is always
    current. Document frequency per term is derived from the inverted index.
    """

    def __init__(self, *, tokenizer: Tokenizer, k1: float, b: float) -> None:
        # Construction goes through ``fit``; this initializer only sets up empty
        # state and parameters. Callers must use ``fit`` (which enforces the
        # non-empty-corpus contract).
        self._tokenizer: Tokenizer = tokenizer
        self._k1 = k1
        self._b = b
        # term -> {doc_id -> term frequency within that doc}
        self._postings: dict[str, dict[ID, int]] = {}
        # doc_id -> token count (document length)
        self._doc_len: dict[ID, int] = {}
        # Insertion-ordered id list, used as the deterministic tie-break key.
        self._order: dict[ID, int] = {}
        self._next_ord = 0
        self._total_len = 0

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def fit(
        cls,
        ids: Sequence[ID],
        texts: Sequence[str],
        *,
        tokenizer: Tokenizer = default_tokenizer,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> "BM25Index":
        """Build an index from parallel ``ids`` and ``texts``.

        Raises :class:`~fusion_recall.contracts.EmptyCorpusError` if the corpus
        is empty and ``ValueError`` if ``ids`` and ``texts`` differ in length.
        """
        if len(ids) != len(texts):
            raise ValueError(
                f"ids and texts must have equal length; got {len(ids)} and {len(texts)}"
            )
        if len(ids) == 0:
            raise EmptyCorpusError("cannot build a BM25 index from an empty corpus")
        index = cls(tokenizer=tokenizer, k1=k1, b=b)
        for doc_id, text in zip(ids, texts):
            index.add(doc_id, text)
        return index

    # ------------------------------------------------------------------ #
    # Mutation (the host keeps the index synced)
    # ------------------------------------------------------------------ #
    def add(self, doc_id: ID, text: str) -> None:
        """Add (or replace) a document and update global statistics.

        Re-adding an existing id replaces its prior contribution so length and
        document-frequency statistics stay correct.
        """
        if doc_id in self._doc_len:
            self.remove(doc_id)
        tokens = self._tokenizer(text)
        tf = Counter(tokens)
        for term, count in tf.items():
            self._postings.setdefault(term, {})[doc_id] = count
        self._doc_len[doc_id] = len(tokens)
        self._total_len += len(tokens)
        self._order[doc_id] = self._next_ord
        self._next_ord += 1

    def remove(self, doc_id: ID) -> None:
        """Remove a document and update global statistics.

        Raises ``KeyError`` if the id is not present — removal of an unknown id
        is a caller error, not a silent no-op (fail-fast).
        """
        if doc_id not in self._doc_len:
            raise KeyError(doc_id)
        for term in list(self._postings.keys()):
            postings = self._postings[term]
            if doc_id in postings:
                del postings[doc_id]
                if not postings:
                    del self._postings[term]
        self._total_len -= self._doc_len.pop(doc_id)
        del self._order[doc_id]

    # ------------------------------------------------------------------ #
    # Query
    # ------------------------------------------------------------------ #
    def __call__(self, query: str, k: int) -> Ranking:
        """Return the top-``k`` documents for ``query`` as a descending Ranking.

        A query with no in-vocabulary terms returns an empty ``Ranking`` rather
        than raising. Documents that score zero (no query-term overlap) are not
        returned. Ties in score are broken by insertion order for determinism.
        """
        n = len(self._doc_len)
        if n == 0:
            return ()
        terms = self._tokenizer(query)
        if not terms:
            return ()
        avgdl = self._total_len / n

        scores: dict[ID, float] = {}
        # Deduplicate query terms but keep BM25's additive-per-distinct-term
        # behavior; a repeated query term does not double-count under standard
        # Okapi BM25 (term presence in the query is what matters here).
        for term in set(terms):
            postings = self._postings.get(term)
            if not postings:
                continue  # out-of-vocabulary term contributes nothing
            df = len(postings)
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for doc_id, tf in postings.items():
                dl = self._doc_len[doc_id]
                denom = tf + self._k1 * (1 - self._b + self._b * dl / avgdl)
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf * (self._k1 + 1) / denom

        if not scores:
            return ()

        # Sort descending by score; break ties by insertion order so the result
        # is deterministic and stable across runs.
        ranked = sorted(
            scores.items(),
            key=lambda kv: (-kv[1], self._order[kv[0]]),
        )
        return tuple((doc_id, float(score)) for doc_id, score in ranked[:k])

    def __len__(self) -> int:
        """The number of documents currently in the index."""
        return len(self._doc_len)
