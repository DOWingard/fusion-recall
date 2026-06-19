"""Evaluation harness for fusion-recall.

Proves objective 2 — a measurable hybrid lift over a dense baseline on a corpus
where lexical signal matters — by isolating the BM25 recall lever and the
cross-encoder precision lever separately.

The dense baseline is a deterministic, lexical-blind stand-in (see
:mod:`eval.dataset`), not a real embedding model: the harness measures the
*levers*, not a specific embedder. Run it with ``python -m eval.run``.
"""

from __future__ import annotations

from eval.metrics import ndcg_at_k, recall_at_k
from eval.run import evaluate, main

__all__ = ["recall_at_k", "ndcg_at_k", "evaluate", "main"]
