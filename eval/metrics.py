"""Ranking metrics for the evaluation harness: recall@k and nDCG@k.

Both take a ranked list of retrieved ids, the set of relevant ids (qrels), and a
cutoff ``k``. Relevance is binary (a doc is relevant or not), so nDCG uses binary
gains. Both are pure and deterministic, and both define the empty-relevant case
as ``0.0`` so the harness never divides by zero.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from fusion_recall.contracts import ID


def recall_at_k(retrieved: Sequence[ID], relevant: set[ID], k: int) -> float:
    """Fraction of relevant ids that appear in the top-``k`` of ``retrieved``.

    ``recall@k = |relevant ∩ retrieved[:k]| / |relevant|``. Returns ``0.0`` when
    there are no relevant ids (the metric is undefined; ``0.0`` keeps it finite).
    """
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    hits = len(top_k & relevant)
    return hits / len(relevant)


def ndcg_at_k(retrieved: Sequence[ID], relevant: set[ID], k: int) -> float:
    """Normalized discounted cumulative gain at ``k`` with binary gains.

    ``DCG@k = Σ_{i=1..k} rel_i / log2(i + 1)`` with ``rel_i ∈ {0, 1}``;
    ``IDCG@k`` is the DCG of the ideal ranking (all relevant docs first, capped at
    ``k``); ``nDCG@k = DCG@k / IDCG@k``. Returns ``0.0`` when there are no
    relevant ids (IDCG would be 0).
    """
    if not relevant:
        return 0.0

    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k]):
        if doc_id in relevant:
            # rank is 1-based: position i (0-based) -> discount log2(i + 2).
            dcg += 1.0 / math.log2(i + 2)

    # Ideal DCG: as many relevant docs as fit in k, each contributing a perfect
    # gain at the earliest possible rank.
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg
