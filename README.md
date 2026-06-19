# fusion-recall

A drop-in hybrid-retrieval layer that wraps any existing dense/embedding retriever and lifts it to full **fusion recall** — parallel BM25 lexical recall, rank fusion, and cross-encoder reranking — returning ranked results plus a score vector suitable for a downstream confidence/entropy gate.

---

## Why

Dense/embedding retrieval has a **recall ceiling**. Relevance that is *lexical* — exact identifiers, function/symbol names, error codes, rare tokens, UUIDs, version strings — gets smeared by embeddings into "nearby concept" space, so the truly relevant document never enters the dense top-N. No amount of reranking recovers a document that was never a candidate.

`fusion-recall` fixes this with the correct topology: run lexical (BM25) **in parallel** with your dense retriever over the full corpus, **fuse** the two ranked lists (so the recall ceiling becomes their union), then apply a **cross-encoder** rerank (the precision engine) on a capped window, and cut to `k`. The cross-encoder also makes aggressive lexical recall *safe* — BM25 widens recall and may inject lexically-similar-but-irrelevant noise; the cross-encoder demotes it.

It is **vector-free by construction**: the library only ever sees query text, document text, and scalar scores/ranks. It never reads, stores, or assumes an embedding vector, so it works with any embedding model, dimensionality, or vector store.

---

## When to use this vs. plain embeddings

Hybrid retrieval is **not a universal win** — whether it beats plain embeddings depends on your corpus. The table below is a real measurement: three public BEIR datasets, 150 sampled judged test queries each, a real `all-MiniLM-L6-v2` bi-encoder as the plain-embeddings baseline, this library's BM25, and the default `ms-marco-MiniLM-L-6-v2` cross-encoder.

| Dataset (retrieval profile) | plain dense | + BM25 (RRF) | + BM25 + cross-encoder |
|---|---|---|---|
| **SciFact** — scientific claims; lexical / entity-heavy | 0.6255 | 0.6882 | **0.6990** |
| **NFCorpus** — medical; mixed | 0.3122 | 0.3389 | **0.3514** |
| **ArguAna** — argument paraphrase; semantic | **0.3845** | 0.3857 | 0.3331 |

*nDCG@10 (graded); **bold** = best in row.*

- **Use full hybrid (`rerank=True`) when relevance is lexical** — exact identifiers, codes, symbol names, rare terms, technical/code corpora. On SciFact, hybrid lifted nDCG@10 **+0.073 (+12%)** over plain embeddings, Recall@10 from 0.764 → **0.842**, and Recall@100 from 0.923 → **0.983**. (Plain dense there even *lost to pure BM25*, 0.673 — the signature of a lexical corpus.)
- **Expect a small, safe gain on mixed corpora** — NFCorpus: **+0.039** nDCG@10.
- **Prefer plain embeddings (or at most `rerank=False`) on semantic / paraphrase corpora** where your embedder already saturates candidate recall. On ArguAna (plain-dense Recall@100 = 0.98 — almost nothing left to recover), BM25 fusion was a statistical tie and the **default cross-encoder *reduced* nDCG@10 by −0.051**: a domain-mismatched reranker reorders good dense candidates *worse*.

**Rule of thumb:** the more your relevant documents hinge on *exact tokens*, the more hybrid helps; the more they hinge on *meaning your embedder already captures*, the less BM25 adds — and a generic cross-encoder can hurt, so validate it on your own data or run `rerank=False`. When plain-dense Recall@100 is already near 1.0, fusion has no recall headroom to add.

**Cost.** BM25 is an in-memory index you keep synced (`add`/`remove`); the cross-encoder adds inference latency proportional to the rerank window (`rerank_top_m`, default 50) per query. Both are opt-out.

---

## Install

```bash
pip install fusion-recall
```

### CPU-only torch (recommended)

`fusion-recall` depends on `sentence-transformers`, which pulls in PyTorch. By default `pip` installs the multi-GB **CUDA** torch wheel. If you do not need a GPU, install the CPU-only torch wheel **first** (or alongside) so the large CUDA build is never downloaded:

```bash
# Install the CPU torch wheel before the package:
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install fusion-recall
```

(or pass `--extra-index-url https://download.pytorch.org/whl/cpu` to a single `pip install` so the CPU wheel is preferred.)

### Offline / baked-model usage

The default cross-encoder (`DEFAULT_CE_MODEL`) is downloaded on first use. To run with no network access, **pre-cache** the model once, then set the standard Hugging Face offline environment variables so loading reads only the local cache:

```bash
# 1. Pre-cache the default model once (online), e.g.:
python -c "from sentence_transformers import CrossEncoder; \
           import fusion_recall as fr; CrossEncoder(fr.DEFAULT_CE_MODEL)"

# 2. Then run fully offline:
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

The cross-encoder loads lazily (importing `fusion_recall` never triggers a torch/model import) and honors these variables, so a host can bake the model into an image and run air-gapped.

---

## Quickstart

```python
from fusion_recall import lift

# 1. Your existing dense retriever: a callable (query, k) -> ranked [(id, score), ...].
#    The embedding model, dimension, and vector store all stay inside this callable.
def dense(query: str, k: int):
    return my_vector_search(query, k)   # returns a descending tuple of (id, score)

# 2. Document text so BM25 can be built and the cross-encoder can read candidates.
corpus = {
    "doc1": "the parser raises error code E_4471 on unterminated input",
    "doc2": "an overview of ranking and search systems",
    # ...
}

# 3. One call returns a working hybrid retriever (full hybrid by default).
retriever = lift(dense, corpus)

result = retriever.retrieve("what raises error code E_4471", k=5)
result.ids        # tuple of the top-k document ids, best first
result.scores     # np.float64 vector, descending, aligned to ids
result.provenance # {"dense","lexical","fused","ce"} score columns aligned to ids
                  # (NaN where a source did not surface that id)
```

`candidate_k` (default 100) is the **per-source** recall budget passed to both the dense and lexical retrievers. The rerank window is clamped to `max(profile.rerank_top_m, k)`, so the returned top-`k` is always single-scale and strictly descending.

Fusion-only mode (no cross-encoder) returns the **fused** scores:

```python
retriever = lift(dense, corpus, rerank=False)   # dense ∥ BM25, RRF fusion, no CE
```

---

## Score hand-off to a confidence/entropy gate

`fusion-recall` is **gate-agnostic**: it ships no gate logic, only `result.scores` (and per-source `result.provenance`). The host decides whether, where, and how to gate. A typical gate turns the scores into a probability distribution with a softmax and reads its entropy (low entropy ⇒ a confident, peaked result; high entropy ⇒ a diffuse, uncertain one):

```python
import numpy as np

def gate_signals(scores: np.ndarray, *, beta: float):
    """Softmax the scores at temperature `beta`, then read normalized entropy."""
    x = scores.astype(np.float64) * beta
    x -= x.max()                       # numerically stable
    p = np.exp(x)
    p /= p.sum()
    n = len(p)
    nz = p[p > 0.0]
    entropy = -np.sum(nz * np.log(nz))
    normalized_entropy = entropy / np.log(n) if n > 1 else 0.0  # in [0, 1]
    top1_mass = float(p.max())
    return normalized_entropy, top1_mass
```

**Choosing `beta` (temperature).** The right temperature depends on the score scale:

| Score source | Scale | Recommended `beta` |
|--------------|-------|--------------------|
| Cross-encoder logits (`rerank=True`, default) | wide (logits) | `beta ≈ 1.0` |
| Cross-encoder probabilities (`ce_sigmoid=True`) | `[0, 1]`, narrow | `beta ≈ 5.0` |
| Fused RRF scores (`rerank=False`) | very narrow (`~1/(k+rank)`) | `beta ≈ 200` |

RRF scores have a small dynamic range, so they need a larger `beta` to separate; cross-encoder logits are already well-spread. Set `ce_sigmoid=True` in the `FusionProfile` to squash cross-encoder scores into `[0, 1]` for the gate.

### Adaptive rerank depth (optional)

The default cross-encoder depth is `max(rerank_top_m, k)`. You can opt into an adaptive depth with a `depth_policy` — a callable that reads the fused scores and returns how deep to rerank (the result is always clamped to `[k, max(rerank_top_m, k)]`). A common policy reranks down to the first large gap in the fused scores:

```python
import numpy as np
from fusion_recall import HybridRetriever

def gap_depth(fused_scores: np.ndarray) -> int:
    """Rerank down to the first big relative drop in fused score (>= 2 deep)."""
    if len(fused_scores) < 2:
        return len(fused_scores)
    drops = fused_scores[:-1] - fused_scores[1:]
    return int(np.argmax(drops)) + 1 + 1   # include the item before the largest gap

retriever = HybridRetriever(dense, lexical=my_bm25, text_of=corpus.__getitem__,
                            reranker=my_ce, depth_policy=gap_depth)
```

---

## Integration contract

`fusion-recall` imports nothing from your application; the dependency points your code → the library. To lift an existing embedding-recall system:

1. **Wrap the dense retriever.** Provide a callable `(query, k) -> [(id, score), ...]` that encapsulates your encode + vector search. The embedding model, dimension, and vector store stay entirely inside it.
2. **Supply document text.** Pass a `corpus` map (or a `text_of` callable) so the library builds and owns a `BM25Index`; or inject your own lexical search as a `LexicalRetriever`. The cross-encoder always needs a text source to fetch candidate text.
3. **Keep the lexical index synced.** If you use the built-in `BM25Index`, call `add(id, text)` / `remove(id)` as your store changes — index synchronization is your responsibility, mirroring how you already maintain your vector index.
4. **Consume the output.** Feed `result.scores` to your confidence/entropy gate. You choose where the gate sits:
   - *gate-before-fusion* — gate on your dense distribution as today; the library only reorders the already-admitted set (conservative);
   - *gate-after-fusion* — gate on `result.scores` so a lexically-recovered document counts toward the gate decision (recalibrate the gate temperature for the fused scale).

   The library supplies both the final `scores` and per-source `provenance` to support either placement.

Every swappable seam is a `typing.Protocol` (`DenseRetriever`, `LexicalRetriever`, `CrossEncoder`, `Fuser`, `TextLookup`, `Tokenizer`, `Normalize`) with a working default where one is possible, so each is independently replaceable.

---

## Public API

```python
from fusion_recall import (
    lift, HybridRetriever,                 # entry points
    BM25Index, default_tokenizer,          # built-in lexical recall
    rrf, weighted_fuse,                    # fusers
    SentenceTransformerCrossEncoder, NoOpCrossEncoder,  # cross-encoders
    FusionProfile, HybridResult,           # value objects
    DEFAULT_PROFILE, DEFAULT_CE_MODEL,     # constants
    # Protocols + exceptions (FusionError, EmptyCorpusError, InvalidProfileError)
)
```

`FusionProfile` tunes fusion + rerank: `method` (`"rrf"` | `"weighted"`), `rrf_k` (RRF damping, default 60), `weights` (for weighted fusion), `rerank_top_m` (cross-encoder window, default 50), and `ce_sigmoid` (squash cross-encoder scores to `[0, 1]`). Weighted fusion ships a built-in min-max normalizer and also accepts any `Normalize` callable by injection — no extra dependency required.

---

## Behavior guarantees

- **Recall union / raised ceiling.** The fused candidate set is the union of the dense and lexical ids; a document only the lexical path finds can reach the final results.
- **No truncation.** The rerank window clamps to `>= k`, so the returned top-`k` is always drawn from one cross-encoder-scored, single-scale, descending window.
- **Fail-open.** A lexical fault degrades to dense order; a cross-encoder fault degrades to fused order. A fault is logged and never raises into your read path.
- **Deterministic pure path.** BM25 + fusion are bit-identical on re-runs; the only nondeterminism is the cross-encoder model, isolated behind its seam.
- **Fail-fast config.** Inconsistent construction (e.g. `rerank=True` with no text source) raises `InvalidProfileError` at build time.

---

## Evaluation

The bundled harness (`python -m eval.run`) proves the lift by isolating each lever separately: dense-only vs dense+BM25 (RRF, no cross-encoder) vs full hybrid (RRF + cross-encoder). It reports the **BM25 lever** (`rrf − dense`) and the **cross-encoder lever** (`hybrid − rrf`) as distinct deltas.

> **Caveat.** The harness's dense baseline is a **deterministic, lexical-blind stand-in**, not a real embedding model. Holding the dense ranking fixed is what makes the two levers separately measurable and reproducible — so the harness measures the *levers*, it is **not** a benchmark of any particular embedder.

For real-embedder numbers on labeled datasets (and the decision rule that follows from them), see [When to use this vs. plain embeddings](#when-to-use-this-vs-plain-embeddings).

---

## Development

```bash
pip install -e ".[dev]"
pytest                       # offline core suite (slow/model tests deselected by default)
pytest -m 'slow or not slow' # also run the model-dependent tests (real cross-encoder)
```

---

## License

MIT — see [LICENSE](LICENSE).
