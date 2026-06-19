"""Reproducible benchmark: plain dense retrieval vs fusion-recall hybrid on BEIR.

Compares four configurations per dataset using a real bi-encoder, this library's
real BM25, and this library's real cross-encoder:

  bm25        - pure BM25Index (lexical baseline)
  dense       - bi-encoder alone ("plain embeddings")
  hybrid_rrf  - lift(dense, corpus, rerank=False)   (dense || BM25, RRF fusion)
  hybrid_ce   - lift(dense, corpus, rerank=True)    (+ cross-encoder rerank)

Metrics: graded nDCG@10 and Recall@10 for all four configs; Recall@100 for
`dense` and `hybrid_rrf` (the candidate-recall ceiling). Averaged over up to
`--max-queries` deterministically sampled judged test queries per dataset.

Datasets are downloaded from the public BEIR mirror on first use and cached
under `--data-dir`. This is a network- and model-heavy script (it downloads the
datasets and the embedding/cross-encoder models, then encodes each corpus on
CPU); it is intentionally NOT part of the offline test suite.

Run:
    python -m benchmarks.beir                      # all three datasets, 150 queries
    python -m benchmarks.beir --datasets scifact --max-queries 50
    python -m benchmarks.beir --out benchmarks/RESULTS.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import tempfile
import time
import urllib.request
import zipfile
from collections.abc import Mapping, Sequence

import numpy as np

from fusion_recall import BM25Index, lift

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
DEFAULT_DATASETS = ["scifact", "nfcorpus", "arguana"]
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # the library default
ENCODE_BATCH = 256


# --------------------------------------------------------------------------- #
# Metrics (graded).
# --------------------------------------------------------------------------- #
def ndcg_at_k(retrieved: Sequence[str], rels: Mapping[str, int], k: int) -> float:
    """Graded nDCG@k.

    DCG@k = sum_{i=1..k} rel_i / log2(i + 1), with rel_i the graded relevance of
    the doc at rank i (0 if unjudged). IDCG@k is the DCG of the ideal ordering.
    Returns 0.0 when the query has no positive judgments.
    """
    positives = [g for g in rels.values() if g > 0]
    if not positives:
        return 0.0
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k]):
        g = rels.get(doc_id, 0)
        if g > 0:
            dcg += g / math.log2(i + 2)  # 0-based rank i -> log2(i + 2)
    ideal = sorted(positives, reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(retrieved: Sequence[str], rels: Mapping[str, int], k: int) -> float:
    """Recall@k = |top-k retrieved ∩ {rel>0}| / |{rel>0}|; 0.0 with no positives."""
    positives = {d for d, g in rels.items() if g > 0}
    if not positives:
        return 0.0
    return len(set(retrieved[:k]) & positives) / len(positives)


def _sanity_check_metrics() -> None:
    """Hand-checked tiny examples; abort the run if the metric code is wrong."""
    rels = {"d1": 3, "d2": 2, "d3": 1}
    assert abs(ndcg_at_k(["d1", "d2", "d3"], rels, 10) - 1.0) < 1e-12
    dcg = 1 / math.log2(2) + 2 / math.log2(3) + 3 / math.log2(4)
    idcg = 3 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(4)
    assert abs(ndcg_at_k(["d3", "d2", "d1"], rels, 10) - dcg / idcg) < 1e-12
    rels2 = {"x": 1}
    assert abs(ndcg_at_k(["z", "x", "y"], rels2, 10) - (1 / math.log2(3))) < 1e-12
    assert ndcg_at_k(["o"] * 10 + ["x"], rels2, 10) == 0.0
    rels3 = {"a": 1, "b": 1}
    assert recall_at_k(["a", "q", "r"], rels3, 3) == 0.5
    assert recall_at_k(["a", "b", "r"], rels3, 3) == 1.0
    assert recall_at_k(["z"] * 10 + ["a", "b"], rels3, 10) == 0.0
    assert recall_at_k(["z"] * 10 + ["a", "b"], rels3, 100) == 1.0
    assert ndcg_at_k(["a"], {"a": 0}, 10) == 0.0 and recall_at_k(["a"], {}, 10) == 0.0


# --------------------------------------------------------------------------- #
# Dataset download + loading (BEIR flat format).
# --------------------------------------------------------------------------- #
def ensure_dataset(name: str, data_dir: str) -> str:
    """Download + unzip the BEIR dataset into ``data_dir`` if not already present.

    Returns the dataset directory (containing corpus.jsonl, queries.jsonl,
    qrels/test.tsv). Raises on download/extract failure (the caller skips).
    """
    base = os.path.join(data_dir, name)
    if os.path.exists(os.path.join(base, "corpus.jsonl")):
        return base
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, f"{name}.zip")
    url = BEIR_URL.format(name=name)
    print(f"[download] {url}", flush=True)
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(data_dir)
    if not os.path.exists(os.path.join(base, "corpus.jsonl")):
        raise FileNotFoundError(f"{name}: corpus.jsonl missing after extract")
    return base


def load_corpus(path: str) -> dict[str, str]:
    """corpus.jsonl -> {doc_id: (title + ' ' + text).strip()}."""
    corpus: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            corpus[str(d["_id"])] = ((d.get("title") or "") + " " + (d.get("text") or "")).strip()
    return corpus


def load_queries(path: str) -> dict[str, str]:
    """queries.jsonl -> {query_id: text}."""
    queries: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            queries[str(d["_id"])] = d["text"]
    return queries


def load_qrels(path: str) -> dict[str, dict[str, int]]:
    """qrels/test.tsv (with header) -> {query_id: {doc_id: graded_rel}} for rel>0."""
    qrels: dict[str, dict[str, int]] = {}
    with open(path, encoding="utf-8") as f:
        header = f.readline()
        assert "query-id" in header, f"unexpected qrels header: {header!r}"
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            qid, did, score = parts[0], parts[1], int(float(parts[2]))
            if score > 0:
                qrels.setdefault(qid, {})[did] = score
    return qrels


# --------------------------------------------------------------------------- #
# Dense retriever over a precomputed normalized corpus matrix.
# --------------------------------------------------------------------------- #
class DenseRetriever:
    """Bi-encoder retriever: cosine (dot of L2-normalized vectors) over the corpus.

    The corpus is encoded once; each query is encoded on call and the top-k
    (id, score) pairs are returned in the library's Ranking shape.
    """

    def __init__(self, model, ids: list[str], doc_matrix: np.ndarray) -> None:
        self._model = model
        self._ids = ids
        self._mat = doc_matrix  # (N, D) float32, L2-normalized rows

    def __call__(self, query: str, k: int):
        q = self._model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False,
        )[0].astype(np.float32)
        sims = self._mat @ q
        k = min(k, len(self._ids))
        top_idx = np.argpartition(-sims, k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        return tuple((self._ids[i], float(sims[i])) for i in top_idx)


# --------------------------------------------------------------------------- #
# Per-dataset benchmark.
# --------------------------------------------------------------------------- #
def run_dataset(name: str, model, *, data_dir: str, max_queries: int, seed: int) -> dict:
    base = ensure_dataset(name, data_dir)
    t0 = time.time()
    corpus = load_corpus(os.path.join(base, "corpus.jsonl"))
    queries = load_queries(os.path.join(base, "queries.jsonl"))
    qrels = load_qrels(os.path.join(base, "qrels", "test.tsv"))

    judged = sorted(
        qid for qid in qrels
        if qid in queries and any(g > 0 for g in qrels[qid].values())
    )
    rng = random.Random(seed)
    sampled = sorted(rng.sample(judged, max_queries) if len(judged) > max_queries else judged)
    n_q = len(sampled)
    print(f"[{name}] corpus={len(corpus)} judged_test_q={len(judged)} using n={n_q}", flush=True)

    ids = list(corpus.keys())
    texts = [corpus[i] for i in ids]
    te = time.time()
    doc_mat = model.encode(
        texts, batch_size=ENCODE_BATCH, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    ).astype(np.float32)
    print(f"[{name}] encoded {len(ids)} docs in {time.time() - te:.1f}s", flush=True)

    dense = DenseRetriever(model, ids, doc_mat)
    bm25 = BM25Index.fit(ids, texts)
    hyb_rrf = lift(dense, corpus, rerank=False)
    hyb_ce = lift(dense, corpus, rerank=True)

    acc = {
        "bm25": {"ndcg10": [], "r10": []},
        "dense": {"ndcg10": [], "r10": [], "r100": []},
        "hybrid_rrf": {"ndcg10": [], "r10": [], "r100": []},
        "hybrid_ce": {"ndcg10": [], "r10": []},
    }
    for qi, qid in enumerate(sampled):
        qtext, rels = queries[qid], qrels[qid]
        bm25_ids = [d for d, _ in bm25(qtext, 10)]
        dense_ids = [d for d, _ in dense(qtext, 10)]
        rrf_ids = list(hyb_rrf.retrieve(qtext, 10, candidate_k=100).ids)
        ce_ids = list(hyb_ce.retrieve(qtext, 10, candidate_k=100).ids)
        for cfg, got in (("bm25", bm25_ids), ("dense", dense_ids),
                         ("hybrid_rrf", rrf_ids), ("hybrid_ce", ce_ids)):
            acc[cfg]["ndcg10"].append(ndcg_at_k(got, rels, 10))
            acc[cfg]["r10"].append(recall_at_k(got, rels, 10))
        # candidate-recall ceiling at k=100 (no cross-encoder needed)
        acc["dense"]["r100"].append(recall_at_k([d for d, _ in dense(qtext, 100)], rels, 100))
        acc["hybrid_rrf"]["r100"].append(
            recall_at_k(list(hyb_rrf.retrieve(qtext, 100, candidate_k=200).ids), rels, 100)
        )
        if (qi + 1) % 25 == 0:
            print(f"[{name}] {qi + 1}/{n_q} queries", flush=True)

    def mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    results = {cfg: {m: mean(v) for m, v in metrics.items()} for cfg, metrics in acc.items()}
    wall = time.time() - t0
    print(f"[{name}] done in {wall:.1f}s", flush=True)
    return {"name": name, "corpus_size": len(corpus), "n_queries": n_q, "wall_s": wall, "results": results}


# --------------------------------------------------------------------------- #
# Reporting.
# --------------------------------------------------------------------------- #
def _fmt(x: float) -> str:
    return f"{x:.4f}" if x == x else "n/a"  # x != x catches NaN


def dataset_table(d: dict) -> str:
    r = d["results"]
    return "\n".join([
        f"### {d['name']}",
        "",
        f"corpus size: {d['corpus_size']} | n_queries: {d['n_queries']} | wall-clock: {d['wall_s']:.1f}s",
        "",
        "| config | nDCG@10 | Recall@10 | Recall@100 |",
        "|---|---|---|---|",
        f"| bm25 | {_fmt(r['bm25']['ndcg10'])} | {_fmt(r['bm25']['r10'])} | - |",
        f"| dense | {_fmt(r['dense']['ndcg10'])} | {_fmt(r['dense']['r10'])} | {_fmt(r['dense']['r100'])} |",
        f"| hybrid_rrf | {_fmt(r['hybrid_rrf']['ndcg10'])} | {_fmt(r['hybrid_rrf']['r10'])} | {_fmt(r['hybrid_rrf']['r100'])} |",
        f"| hybrid_ce | {_fmt(r['hybrid_ce']['ndcg10'])} | {_fmt(r['hybrid_ce']['r10'])} | - |",
        "",
    ])


def report(all_results: list[dict], *, model: str, max_queries: int, seed: int) -> str:
    header = (
        "# fusion-recall vs plain dense retrieval — BEIR benchmark\n\n"
        f"Dense bi-encoder: `{model}` (CPU, normalize_embeddings=True). "
        f"Cross-encoder: `{CE_MODEL}` (library default). Lexical: library `BM25Index`. "
        "RRF k=60, CE window rerank_top_m=50. "
        f"Up to {max_queries} judged test queries sampled deterministically (seed={seed}). "
        "nDCG@10 is graded; Recall@100 retrieved at k=100 (candidate_k=200 for hybrid_rrf).\n\n"
        "Configs: **bm25** pure BM25; **dense** bi-encoder alone; "
        "**hybrid_rrf** `lift(dense, corpus, rerank=False)`; "
        "**hybrid_ce** `lift(dense, corpus, rerank=True)`.\n\n"
    )
    body = "\n".join(dataset_table(d) for d in all_results) if all_results else "_No datasets completed._\n"
    return header + body


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Reproduce the fusion-recall BEIR benchmark.")
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    p.add_argument("--max-queries", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--data-dir", default=os.path.join(tempfile.gettempdir(), "fusion-recall-beir"))
    p.add_argument("--out", default=None, help="write the markdown report to this path (else stdout only)")
    args = p.parse_args(argv)

    _sanity_check_metrics()
    print(f"[load] dense model: {args.model}", flush=True)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model, device="cpu")

    all_results: list[dict] = []
    for name in args.datasets:
        try:
            all_results.append(run_dataset(
                name, model, data_dir=args.data_dir, max_queries=args.max_queries, seed=args.seed,
            ))
        except Exception as exc:  # noqa: BLE001 — robustness over completeness
            print(f"[skip] {name}: {type(exc).__name__}: {exc}", flush=True)

    md = report(all_results, model=args.model, max_queries=args.max_queries, seed=args.seed)
    print("\n" + "=" * 70 + "\n" + md + "=" * 70, flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"[written] {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
