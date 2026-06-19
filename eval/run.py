"""Evaluation runner: compute recall@k / nDCG@k for the three configurations and
print the comparison table plus the two per-lever deltas.

The three configurations isolate each lever:

* ``dense``  — the deterministic lexical-blind dense stand-in alone.
* ``rrf``    — dense ∥ BM25, fused by Reciprocal Rank Fusion, **no** cross-encoder
  (``lift(..., rerank=False)``). The ``rrf - dense`` delta isolates the BM25
  recall lever.
* ``hybrid`` — RRF fusion **plus** the default cross-encoder
  (``lift(..., rerank=True)``). The ``hybrid - rrf`` delta isolates the
  cross-encoder precision lever.

The ``dense`` and ``rrf`` paths are pure and reproducible (no model). The
``hybrid`` path loads the default cross-encoder and so requires the model to be
available (cached / offline-bakeable).
"""

from __future__ import annotations

from typing import Literal

from fusion_recall.hybrid import HybridRetriever, lift

from eval.dataset import Query, dense_standin, load_dataset
from eval.metrics import ndcg_at_k, recall_at_k

Config = Literal["dense", "rrf", "hybrid"]


def _retrieve_ids(
    config: Config, query: Query, corpus, k: int
) -> list:
    """Return the ranked ids a config produces for one query, cut to ``k``.

    ``candidate_k`` is set to the full corpus size so the recall ceiling is the
    whole corpus for every config — the eval measures ordering/recall quality,
    not a candidate-budget artifact.
    """
    candidate_k = len(corpus)
    if config == "dense":
        ranking = dense_standin(query.text, k)
        return [doc_id for doc_id, _ in ranking]

    if config == "rrf":
        retriever: HybridRetriever = lift(dense_standin, corpus, rerank=False)
    elif config == "hybrid":
        retriever = lift(dense_standin, corpus, rerank=True)
    else:  # pragma: no cover - guarded by the Literal type
        raise ValueError(f"unknown config: {config!r}")

    result = retriever.retrieve(query.text, k, candidate_k=candidate_k)
    return list(result.ids)


def evaluate(config: Config, *, k: int) -> dict[str, float]:
    """Mean recall@k and nDCG@k over the dataset for one configuration.

    Returns ``{"recall@k": float, "ndcg@k": float}`` averaged across all queries.
    The ``dense`` and ``rrf`` configs are deterministic; ``hybrid`` depends on the
    cross-encoder model.
    """
    corpus, queries = load_dataset()
    recalls: list[float] = []
    ndcgs: list[float] = []
    for query in queries:
        retrieved = _retrieve_ids(config, query, corpus, k)
        relevant = set(query.relevant)
        recalls.append(recall_at_k(retrieved, relevant, k))
        ndcgs.append(ndcg_at_k(retrieved, relevant, k))

    n = len(queries)
    return {
        "recall@k": sum(recalls) / n,
        "ndcg@k": sum(ndcgs) / n,
    }


def main() -> None:
    """Print the three-config comparison table and the two per-lever deltas."""
    k = 5
    dense = evaluate("dense", k=k)
    rrf = evaluate("rrf", k=k)
    hybrid = evaluate("hybrid", k=k)

    print(f"fusion-recall evaluation (k={k})")
    print("-" * 48)
    print(f"{'config':<10}{'recall@k':>14}{'ndcg@k':>14}")
    print("-" * 48)
    for name, scores in (("dense", dense), ("rrf", rrf), ("hybrid", hybrid)):
        print(f"{name:<10}{scores['recall@k']:>14.4f}{scores['ndcg@k']:>14.4f}")
    print("-" * 48)

    # Each lever's contribution, reported separately.
    bm25_recall = rrf["recall@k"] - dense["recall@k"]
    bm25_ndcg = rrf["ndcg@k"] - dense["ndcg@k"]
    ce_recall = hybrid["recall@k"] - rrf["recall@k"]
    ce_ndcg = hybrid["ndcg@k"] - rrf["ndcg@k"]
    print(f"BM25 lever (rrf - dense):   recall {bm25_recall:+.4f}   ndcg {bm25_ndcg:+.4f}")
    print(f"CE   lever (hybrid - rrf):  recall {ce_recall:+.4f}   ndcg {ce_ndcg:+.4f}")


if __name__ == "__main__":
    main()
