#!/usr/bin/env python3
"""
Semantic Odometry -- learned frame transport for bi-encoder retrieval (v0.1)
============================================================================

The feasibility probe showed that a frozen retrieval encoder (E5) carries a
weak but genuine negation frame: a learned subspace beat a random one on
NevIR, while the absolute gains were small. This script tests the stronger
claim that the frame should be *learned* under the paradigm's structural
constraints, and that this structural prior transfers across negation
benchmarks where ordinary fine-tuning does not (van den Elsen et al., SIGIR
2025, found NevIR fine-tuning does not reliably transfer to ExcluIR).

Model
-----
  token states x_i from the encoder (last layer)
  parity p_i in {0,1} from the cue/scope detector (nested cues compose, so
  double negation cancels by construction)
  transported state  x_i' = x_i - c * p_i * U U^T x_i     U: d x k orthonormal
  sentence embedding = L2-normalised mean of x_i'
  (c < 0 amplifies the frame component, c = 2 reflects it; c is learned.)

Conditions (--modes)
--------------------
  zeroshot       base encoder, no training (reference)
  plain_ft       fine-tune encoder, no frame module   (the baseline that
                 failed to transfer in the SIGIR 2025 reproduction)
  frame          fine-tune encoder + learned scoped frame   (the proposal)
  frame_randomU  fine-tune encoder + scoped frame with a FIXED random U,
                 c still learned   (control: rules out "scope marking")
  global_frame   fine-tune encoder + learned frame applied to the whole
                 pooled vector of texts with an odd cue count (RISE-like)
  frame_only     encoder frozen, only U and c trained (cheap readout)

Evaluation (best epoch chosen on NevIR validation only)
-------------------------------------------------------
  NevIR test     pairwise accuracy (random = 0.25)
  ExcluIR        zero-shot transfer: RR (d+ ranked above d-, random = 0.5)
                 and pooled dR@1 / dMRR@10 over the benchmark's own documents
  SciFact (BEIR) nDCG@10, a no-regression check on ordinary retrieval

Usage
-----
  # quick check that everything runs (tiny random model, no downloads)
  python so_train.py --smoke_test

  # main experiment (GPU strongly recommended; see notes in the reply)
  python so_train.py --modes zeroshot plain_ft frame frame_randomU \
      --excluir test_manual_final.json --excluir_full --scifact --seeds 1 2 3

Requires semantic_odometry_probe.py in the same folder (cue/scope detector).
"""

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import semantic_odometry_probe as _sop  # noqa: E402
from semantic_odometry_probe import n_cues, token_parities, synthetic_nevir  # noqa: E402

# Exclusion markers added to the cue lexicon for training/evaluation. Fixed on
# 2026-10-21, BEFORE any ExcluIR result was seen, from general English usage
# (not fitted to the benchmark), so the zero-shot transfer claim stays clean.
# The probe script keeps its original lexicon for reproducibility.
# Multi-word markers ("other than", "apart from", "rather than") are not
# covered: the detector is word-level and "other"/"apart"/"rather" alone are
# too ambiguous. Swapping in a proper cue/scope model (e.g. NCSR) fixes this.
EXCLUSION_CUES = {"excluding", "exclude", "excludes", "excluded", "except",
                  "besides", "instead"}


def use_exclusion_cues(on=True):
    if on:
        _sop.CUES |= EXCLUSION_CUES
        _sop.BOUNDARIES.discard("except")


# ---------------------------------------------------------------------------
# Extended scope detector -- used ONLY by the pre-registered reflection test
# (--reflect_eval), never in training, so trained models are unaffected.
# Adds general-English multi-word exclusion markers that the word-level
# lexicon cannot express. Fixed 2026-10-26 BEFORE any reflection result was
# seen; not derived from ExcluIR.
# ---------------------------------------------------------------------------
EXT_PHRASES = {("apart", "from"), ("aside", "from"), ("other", "than"),
               ("rather", "than"), ("not", "including")}


def word_parities_ext(text):
    ms = list(_sop.WORD_RE.finditer(text))
    out, count, i = [], 0, 0
    while i < len(ms):
        w = ms[i].group(0)
        lw = w.lower()
        nxt = ms[i + 1].group(0).lower() if i + 1 < len(ms) else None
        if (lw, nxt) in EXT_PHRASES:
            for m in (ms[i], ms[i + 1]):
                out.append((m.start(), m.end(), count % 2, True))
            count += 1
            i += 2
            continue
        if lw in _sop.BOUNDARIES:
            count = 0
            out.append((ms[i].start(), ms[i].end(), 0, False))
        elif _sop.is_cue(w):
            out.append((ms[i].start(), ms[i].end(), count % 2, True))
            count += 1
        else:
            out.append((ms[i].start(), ms[i].end(), count % 2, False))
        i += 1
    return out


def _map_parities(words, offsets):
    par = np.zeros(len(offsets), dtype=np.int8)
    wi = 0
    for ti, (s, e) in enumerate(offsets):
        if e <= s:
            continue
        while wi < len(words) and words[wi][1] <= s:
            wi += 1
        if wi < len(words) and words[wi][0] <= s < words[wi][1]:
            par[ti] = words[wi][2]
    return par


def token_parities_ext(text, offsets):
    return _map_parities(word_parities_ext(text), offsets)


def scope_coverage(texts):
    """Fraction of texts with at least one in-scope word, under the training
    detector and under the extended (reflection-test) detector."""
    base = np.mean([any(p for a, b, p, c in _sop.word_parities(t)) for t in texts])
    ext = np.mean([any(p for a, b, p, c in word_parities_ext(t)) for t in texts])
    return float(base), float(ext)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class FrameEncoder(nn.Module):
    def __init__(self, backbone, tokenizer, mode, k=8, c_init=-2.0,
                 max_len=256, e5_prefix=False, seed=0):
        super().__init__()
        self.backbone, self.tok, self.mode = backbone, tokenizer, mode
        self.max_len, self.e5_prefix = max_len, e5_prefix
        d = backbone.config.hidden_size
        g = torch.Generator().manual_seed(seed)
        U0 = torch.randn(d, k, generator=g)
        if mode == "frame_randomU":
            q, _ = torch.linalg.qr(U0)
            self.register_buffer("U_fixed", q)
        else:
            self.U_raw = nn.Parameter(U0)
        self.c = nn.Parameter(torch.tensor(float(c_init)))
        if mode in ("zeroshot", "frame_only"):
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        # Inference-only override for the reflection test: {"U": d x k, "c": 2.0}.
        # When set, QUERIES get X - c * par_ext * U U^T X (extended scope) in
        # place of any trained transport; documents are left as trained.
        self.reflect = None
        # Inference-only CONTENT readout ("aboutness" score): when set to a
        # d x k matrix U, BOTH queries and documents are embedded as
        # normalize((I - U U^T) h_raw), i.e. the encoder output with the frame
        # subspace projected out, which is polarity-blind by construction.
        self.content_U = None

    @property
    def uses_frame(self):
        return self.mode in ("frame", "frame_randomU", "global_frame", "frame_only")

    def U(self):
        if self.mode == "frame_randomU":
            return self.U_fixed
        # QR runs on CPU: aten::linalg_qr is not implemented on MPS (torch 2.2).
        # U is tiny (d x k), and autograd flows through the device transfers.
        q, _ = torch.linalg.qr(self.U_raw.cpu())
        return q.to(self.U_raw.device)

    def prefix(self, texts, role):
        if not self.e5_prefix:
            return list(texts)
        p = "query: " if role == "query" else "passage: "
        return [p + t for t in texts]

    def forward(self, texts, role, return_raw=False):
        """Returns the transported, L2-normalised embedding. With return_raw,
        also returns the encoder's own pooled embedding BEFORE any frame
        transport (used by the auxiliary frame loss and the probe init)."""
        texts = self.prefix(texts, role)
        enc = self.tok(texts, padding=True, truncation=True, max_length=self.max_len,
                       return_offsets_mapping=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping")
        dev = next(self.backbone.parameters()).device
        enc = {k: v.to(dev) for k, v in enc.items()}
        X = self.backbone(**enc).last_hidden_state              # B x T x d
        mask = enc["attention_mask"].unsqueeze(-1).float()
        h_raw = (X * mask).sum(1) / mask.sum(1).clamp(min=1.0)

        if self.content_U is not None:
            U = self.content_U.to(dev, h_raw.dtype)
            hc = h_raw - (h_raw @ U) @ U.T
            return (F.normalize(hc, dim=-1), h_raw) if return_raw else F.normalize(hc, dim=-1)

        if self.reflect is not None and role == "query":
            par = torch.zeros(X.shape[:2], device=dev)
            for b, t in enumerate(texts):
                offs = [tuple(o) for o in offsets[b].tolist()]
                par[b] = torch.as_tensor(token_parities_ext(t, offs), dtype=torch.float32)
            U = self.reflect["U"].to(dev, X.dtype)
            Xr = X - self.reflect["c"] * (par.unsqueeze(-1) * mask) * ((X @ U) @ U.T)
            h = (Xr * mask).sum(1) / mask.sum(1).clamp(min=1.0)
            return (F.normalize(h, dim=-1), h_raw) if return_raw else F.normalize(h, dim=-1)

        if self.mode in ("frame", "frame_randomU", "frame_only"):
            par = torch.zeros(X.shape[:2], device=dev)
            for b, t in enumerate(texts):
                offs = [tuple(o) for o in offsets[b].tolist()]
                par[b] = torch.as_tensor(token_parities(t, offs), dtype=torch.float32)
            U = self.U()
            X = X - self.c * (par.unsqueeze(-1) * mask) * ((X @ U) @ U.T)

        h = (X * mask).sum(1) / mask.sum(1).clamp(min=1.0)

        if self.mode == "global_frame":
            odd = torch.tensor([n_cues(t) % 2 for t in texts], dtype=torch.float32,
                               device=dev).unsqueeze(-1)
            U = self.U()
            h = h - self.c * odd * ((h @ U) @ U.T)
        if return_raw:
            return F.normalize(h, dim=-1), h_raw
        return F.normalize(h, dim=-1)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_nevir():
    from datasets import load_dataset
    ds = load_dataset("orionweller/NevIR")
    return {sp: [(r["q1"], r["q2"], r["doc1"], r["doc2"]) for r in ds[sp]]
            for sp in ("train", "validation", "test")}


def nevir_triplets(rows):
    out = []
    for q1, q2, d1, d2 in rows:
        out.append((q1, d1, d2))
        out.append((q2, d2, d1))
    return out


def load_general(name, n, seed, val_n=0):
    """(query, positive) pairs for in-batch-negative training, to guard
    against regression on ordinary retrieval, plus a DISJOINT validation
    split (val_n pairs) for model selection without touching any test set.
    Returns (train_pairs, val_pairs)."""
    if not name or (n <= 0 and val_n <= 0):
        return [], []
    from datasets import load_dataset
    ds = load_dataset(name, split="train")
    cols = ds.column_names
    qc = next(c for c in ("query", "question", "anchor", "sentence1") if c in cols)
    pc = next(c for c in ("answer", "positive", "passage", "sentence2", "pos") if c in cols)
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    train = [(ds[i][qc], ds[i][pc]) for i in idx[:n]]
    val = [(ds[i][qc], ds[i][pc]) for i in idx[n:n + val_n]]
    return train, val


def _pick(d, names):
    for n in names:
        if n in d and d[n]:
            v = d[n]
            if isinstance(v, list):
                v = v[0]
            if isinstance(v, dict):
                v = v.get("text") or v.get("contents") or " ".join(map(str, v.values()))
            return str(v)
    return None


def load_excluir(path, corpus_path=None):
    """Load the ExcluIR benchmark as (query, positive_text, negative_text).

    The released test_manual_final.json stores corpus indices, not text:
      {"question0": ..., "RQ_rewrite": <exclusionary query>,
       "corpus_sub_index": [neg_idx, pos_idx]}
    The authors' bi_eval.py (fields "ExcluQ"/"index" in their older naming)
    takes index[1:] as the POSITIVE and index[:1] as the NEGATIVE document,
    so the second index is positive. Returns (rows, corpus, id_pairs) where
    corpus/id_pairs are None for text-format files.
    """
    with open(path, encoding="utf-8") as f:
        txt = f.read().strip()
    rows = json.loads(txt) if txt.startswith("[") else [json.loads(l) for l in txt.splitlines() if l.strip()]
    if isinstance(rows, dict):
        rows = next(v for v in rows.values() if isinstance(v, list))

    idx_key = next((k for k in ("corpus_sub_index", "index") if k in rows[0]), None)
    if idx_key:
        q_key = next(k for k in ("RQ_rewrite", "ExcluQ") if k in rows[0])
        corpus_path = corpus_path or os.path.join(os.path.dirname(os.path.abspath(path)),
                                                  "corpus.json")
        with open(corpus_path, encoding="utf-8") as f:
            corpus = json.load(f)
        out, ids = [], []
        for r in rows:
            neg_i, pos_i = int(r[idx_key][0]), int(r[idx_key][1])
            out.append((r[q_key], corpus[pos_i], corpus[neg_i]))
            ids.append((pos_i, neg_i))
        q, p, n = out[0]
        print(f"  ExcluIR: {len(out)} queries; corpus {len(corpus)} docs "
              f"(query field '{q_key}', indices '{idx_key}', second index = positive)")
        print(f"  sanity check, first example:\n    query: {q[:150]}\n"
              f"    POSITIVE: {p[:120].replace(chr(10), ' | ')}\n"
              f"    EXCLUDED: {n[:120].replace(chr(10), ' | ')}")
        return out, corpus, ids
    qn = ("question", "query", "exclusionary_query", "q")
    pn = ("positive", "pos", "positive_doc", "pos_doc", "positive_passage", "doc_pos", "d_pos", "doc1")
    nn_ = ("negative", "neg", "negative_doc", "neg_doc", "negative_passage", "doc_neg", "d_neg", "doc2")
    out = []
    for r in rows:
        q, p, n = _pick(r, qn), _pick(r, pn), _pick(r, nn_)
        if q and p and n:
            out.append((q, p, n))
    if not out:
        raise ValueError(f"Could not find query/positive/negative fields in {path}. "
                         f"Keys of first row: {list(rows[0].keys())}")
    print(f"  ExcluIR: {len(out)} instances loaded (fields auto-detected from "
          f"{list(rows[0].keys())})")
    return out, None, None


def load_beir(name, splits=("test",)):
    """Load a BEIR dataset from the HF hub as (docs, queries, relevance), using
    the first split in `splits` that the qrels repository provides."""
    from datasets import load_dataset
    corpus = load_dataset(f"BeIR/{name}", "corpus")["corpus"]
    queries = load_dataset(f"BeIR/{name}", "queries")["queries"]
    qd = load_dataset(f"BeIR/{name}-qrels")
    split = next((sp for sp in splits if sp in qd), None)
    if split is None:
        raise ValueError(f"BeIR/{name}-qrels has splits {list(qd)}; none of {splits}")
    qrels = qd[split]
    docs = {r["_id"]: (r["title"] + " " + r["text"]).strip() for r in corpus}
    rel = {}
    for r in qrels:
        if r["score"] > 0:
            rel.setdefault(str(r["query-id"]), set()).add(str(r["corpus-id"]))
    qs = {r["_id"]: r["text"] for r in queries if r["_id"] in rel}
    print(f"  BEIR {name} [{split}]: {len(qs)} queries, {len(docs)} documents")
    return docs, qs, rel


def load_scifact():
    from datasets import load_dataset
    corpus = load_dataset("BeIR/scifact", "corpus")["corpus"]
    queries = load_dataset("BeIR/scifact", "queries")["queries"]
    qrels = load_dataset("BeIR/scifact-qrels")["test"]
    docs = {r["_id"]: (r["title"] + " " + r["text"]).strip() for r in corpus}
    rel = {}
    for r in qrels:
        if r["score"] > 0:
            rel.setdefault(str(r["query-id"]), set()).add(str(r["corpus-id"]))
    qs = {r["_id"]: r["text"] for r in queries if r["_id"] in rel}
    return docs, qs, rel

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def embed(model, texts, role, bs):
    model.eval()
    out = []
    for i in range(0, len(texts), bs):
        out.append(model(texts[i:i + bs], role).float().cpu())
    return torch.cat(out) if out else torch.empty(0)


def eval_nevir(model, rows, bs):
    q1 = embed(model, [r[0] for r in rows], "query", bs)
    q2 = embed(model, [r[1] for r in rows], "query", bs)
    d1 = embed(model, [r[2] for r in rows], "doc", bs)
    d2 = embed(model, [r[3] for r in rows], "doc", bs)
    s = lambda a, b: (a * b).sum(1)
    corr = ((s(q1, d1) > s(q1, d2)) & (s(q2, d2) > s(q2, d1))).numpy()
    return float(corr.mean()), corr


@torch.no_grad()
def excluir_official(model, Q, corpus, ids, bs):
    """Authors' protocol (bi_eval.py + utils.py): retrieve top-10 from the full
    corpus. R@1 / MRR@10 for the positive, deltas vs the excluded document, and
    their RR: positive ranked above excluded within the top 10, or positive in
    the top 10 while the excluded one is not. Reported as fractions."""
    C = embed(model, corpus, "doc", bs)
    tops = []
    for i in range(0, Q.shape[0], 256):
        tops.append(torch.topk(Q[i:i + 256] @ C.T, k=10, dim=1).indices)
    top = torch.cat(tops).tolist()
    r1p = r1n = mp = mn = rr = 0.0
    for t, (pi, ni) in zip(top, ids):
        r1p += t[0] == pi
        r1n += t[0] == ni
        mp += 1.0 / (t.index(pi) + 1) if pi in t else 0.0
        mn += 1.0 / (t.index(ni) + 1) if ni in t else 0.0
        if pi in t and ni in t:
            rr += t.index(pi) < t.index(ni)
        elif pi in t:
            rr += 1
    n = len(ids)
    return {"off_R@1": r1p / n, "off_MRR@10": mp / n, "off_dR@1": (r1p - r1n) / n,
            "off_dMRR@10": (mp - mn) / n, "off_RR": rr / n}


def eval_excluir(model, rows, bs, corpus=None, ids=None, full_corpus=False):
    q = embed(model, [r[0] for r in rows], "query", bs)
    pos = embed(model, [r[1] for r in rows], "doc", bs)
    neg = embed(model, [r[2] for r in rows], "doc", bs)
    rr = ((q * pos).sum(1) > (q * neg).sum(1)).numpy()
    official = excluir_official(model, q, corpus, ids, bs) \
        if (full_corpus and corpus is not None) else {}
    # pooled ranking over the benchmark's own unique documents
    pool, ix = [], {}
    for _, p, n in rows:
        for t in (p, n):
            if t not in ix:
                ix[t] = len(pool)
                pool.append(t)
    P = embed(model, pool, "doc", bs)
    S = q @ P.T
    order = torch.argsort(S, dim=1, descending=True)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(S.shape[1]).expand_as(order))
    rp = rank[torch.arange(len(rows)), torch.tensor([ix[r[1]] for r in rows])] + 1
    rn = rank[torch.arange(len(rows)), torch.tensor([ix[r[2]] for r in rows])] + 1
    mrr = lambda r: torch.where(r <= 10, 1.0 / r.float(), torch.zeros_like(r, dtype=torch.float))
    res = {
        "RR": float(rr.mean()),                 # pairwise, random = 0.5
        "dR@1": float((rp == 1).float().mean() - (rn == 1).float().mean()),
        "dMRR@10": float(mrr(rp).mean() - mrr(rn).mean()),
        "pool_size": len(pool),
    }
    res.update(official)
    return res, rr


def eval_general_val(model, pairs, bs):
    """MRR@10 of each query's own positive among all validation positives
    (in-pool retrieval). A cheap, held-out proxy for general retrieval quality,
    used to detect forgetting during training."""
    if not pairs:
        return float("nan")
    Q = embed(model, [q for q, _ in pairs], "query", bs)
    P = embed(model, [p for _, p in pairs], "doc", bs)
    S = Q @ P.T
    ranks = (S > S.diag().unsqueeze(1)).sum(1) + 1
    return float(torch.where(ranks <= 10, 1.0 / ranks.float(),
                             torch.zeros_like(ranks, dtype=torch.float)).mean())


def freeze_lower_layers(backbone, n_top):
    """Freeze embeddings and all transformer layers except the top n_top.
    Keeps the encoder's general retrieval ability in the frozen layers."""
    layers = None
    for path in ("encoder.layer", "layers", "transformer.layer"):
        obj = backbone
        try:
            for a in path.split("."):
                obj = getattr(obj, a)
            layers = obj
            break
        except AttributeError:
            continue
    if layers is None:
        raise ValueError("could not find the transformer layer list for --train_top_layers")
    for p in backbone.parameters():
        p.requires_grad_(False)
    for layer in list(layers)[-n_top:]:
        for p in layer.parameters():
            p.requires_grad_(True)
    n_tr = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in backbone.parameters())
    return n_tr, n_all


def eval_ndcg10(model, docs, queries, rel, bs, per_query=False):
    dids = list(docs)
    D = embed(model, [docs[d] for d in dids], "doc", bs)
    qids = list(queries)
    Q = embed(model, [queries[q] for q in qids], "query", bs)
    top = torch.topk(Q @ D.T, k=min(10, len(dids)), dim=1).indices
    scores = []
    for i, qid in enumerate(qids):
        r = rel[qid]
        dcg = sum(1.0 / math.log2(j + 2) for j, di in enumerate(top[i].tolist())
                  if dids[di] in r)
        idcg = sum(1.0 / math.log2(j + 2) for j in range(min(len(r), 10)))
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    if per_query:
        return float(np.mean(scores)), qids, scores
    return float(np.mean(scores))

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def contrastive_loss(model, batch, tau, hard):
    if hard:
        q, p, n = zip(*batch)
        Q = model(list(q), "query")
        D = torch.cat([model(list(p), "doc"), model(list(n), "doc")])
    else:
        q, p = zip(*batch)
        Q = model(list(q), "query")
        D = model(list(p), "doc")
    logits = Q @ D.T / tau
    return F.cross_entropy(logits, torch.arange(len(batch), device=logits.device))


def frame_aux_loss(model, batch):
    """Structural pressure for the paradigm's claim: the encoder's own change
    between a NevIR document and its negated twin should lie in the frame
    subspace U, so negation must pass through the frame instead of being
    absorbed elsewhere. Loss = fraction of the (raw, pre-transport) pair
    difference lying OUTSIDE span(U); 0 = all of it inside."""
    _, p, n = zip(*batch)
    _, Rp = model(list(p), "doc", return_raw=True)
    _, Rn = model(list(n), "doc", return_raw=True)
    diff = F.normalize(Rp, dim=-1) - F.normalize(Rn, dim=-1)
    U = model.U()
    resid = diff - (diff @ U) @ U.T
    return ((resid ** 2).sum(1) / (diff ** 2).sum(1).clamp(min=1e-8)).mean()


@torch.no_grad()
def probe_init_U(model, nevir_train, k, bs, cache_path=None):
    """Initialise U from the frozen encoder's own negation geometry: the
    offset-aware frame fit of the feasibility probe on NevIR training pairs
    (pre-transport pooled embeddings). Training data only, so no leakage."""
    from semantic_odometry_probe import fit_frame, orient_pairs
    if cache_path and os.path.exists(cache_path):
        return np.load(cache_path)
    pairs_q, pairs_d = [], []
    for q1, q2, d1, d2 in nevir_train:
        pairs_q += orient_pairs([(q1, q2)])
        pairs_d += orient_pairs([(d1, d2)])
    model.eval()

    def raw(texts, role):
        out = []
        for i in range(0, len(texts), bs):
            out.append(model(texts[i:i + bs], role, return_raw=True)[1].float().cpu())
        return torch.cat(out).numpy()

    A = np.vstack([raw([a for a, _ in pairs_d], "doc"), raw([a for a, _ in pairs_q], "query")])
    B = np.vstack([raw([b for _, b in pairs_d], "doc"), raw([b for _, b in pairs_q], "query")])
    U, c, _ = fit_frame(A, B, k)
    print(f"    probe init: U fitted on {len(A)} train pairs (k={k}, fitted c={c:.3f})")
    if cache_path:
        np.save(cache_path, U)
    return U


def content_eval(args, model, mode, data, res, per_instance):
    """Pre-registered content-readout test (fixed 2026-10-27, before results).

    Hypothesis: the frame subspace U carries orientation (polarity); projecting
    it out of BOTH query and document embeddings yields a polarity-blind
    'aboutness' score that recovers topical retrieval (SciFact), while the
    full readout keeps the negation/exclusion gains. U = the model's own frame
    (learned U for frame models; for models without a frame, U refitted by the
    probe procedure on that model's own NevIR TRAIN pairs). Control: random
    orthonormal U of equal rank. Inference only; nothing is tuned.
    """
    if model.uses_frame:
        U = model.U().detach().float()
        src = "model frame"
    else:
        U = torch.as_tensor(probe_init_U(model, data["nevir"]["train"], args.k,
                                         args.eval_batch_size), dtype=torch.float32)
        src = "probe refit on this model (NevIR train)"
    g = torch.Generator().manual_seed(4242)
    U_rand, _ = torch.linalg.qr(torch.randn(U.shape[0], U.shape[1], generator=g))
    print(f"    content readout (frame subspace projected out; U from {src}):")
    for name, UU in (("content", U), ("content_randU", U_rand)):
        model.content_U = UU
        line = f"      {name:<14}"
        if data.get("scifact"):
            m, qids, sc = eval_ndcg10(model, *data["scifact"], args.eval_batch_size,
                                      per_query=True)
            sc = np.array(sc)
            neg = np.array(per_instance.get("scifact_negated", [False] * len(sc)), dtype=bool)
            res[f"{name}_scifact_ndcg10"] = m
            res[f"{name}_scifact_ndcg10_negated"] = float(sc[neg].mean()) if neg.any() else float("nan")
            res[f"{name}_scifact_ndcg10_affirmative"] = float(sc[~neg].mean())
            per_instance[f"scifact_ndcg_q_{name}"] = sc.tolist()
            line += (f" SciFact {m:.3f} (neg {res[f'{name}_scifact_ndcg10_negated']:.3f}, "
                     f"aff {res[f'{name}_scifact_ndcg10_affirmative']:.3f})")
        nv, nvc = eval_nevir(model, data["nevir"]["test"], args.eval_batch_size)
        res[f"{name}_nevir_test"] = nv
        per_instance[f"nevir_test_{name}"] = nvc.tolist()
        line += f" | NevIR {nv:.3f}"
        if data.get("excluir"):
            ex, rr = eval_excluir(model, data["excluir"][0], args.eval_batch_size)
            res[f"{name}_excluir_RR"] = ex["RR"]
            per_instance[f"excluir_rr_{name}"] = rr.tolist()
            line += f" | ExcluIR RR {ex['RR']:.3f}"
        model.content_U = None
        print(line)


def reflect_eval(args, model, mode, data, res, per_instance):
    """Pre-registered reflection test (fixed 2026-10-26, before any result).

    Hypothesis: exclusion is a frame REFLECTION of the scoped (excluded)
    content, not an amplification. Operator: c = +2 (Householder reflection
    on span(U)), applied to QUERY scope only, extended scope detector,
    inference only, no tuning. U = the model's own frame subspace: learned U
    for frame modes, fixed U for frame_randomU, and for models without a frame
    U refitted by the probe procedure on that model's own NevIR TRAIN pairs.
    Control: the same operator with a random orthonormal U of equal rank.
    """
    from scipy.stats import binomtest
    rows_x = data["excluir"][0]
    if model.uses_frame:
        U = model.U().detach()
        src = "model frame"
    else:
        k = args.k
        U = torch.as_tensor(probe_init_U(model, data["nevir"]["train"], k,
                                         args.eval_batch_size), dtype=torch.float32)
        src = "probe refit on this model (NevIR train)"
    g = torch.Generator().manual_seed(12345)
    U_rand, _ = torch.linalg.qr(torch.randn(U.shape[0], U.shape[1], generator=g))
    base_rr = np.array(per_instance["excluir_rr"], dtype=bool)
    out = {}
    for name, UU in (("reflect", U), ("reflect_randU", U_rand)):
        model.reflect = {"U": UU, "c": float(args.reflect_c)}
        ex, rr = eval_excluir(model, rows_x, args.eval_batch_size)
        nv, nvc = eval_nevir(model, data["nevir"]["test"], args.eval_batch_size)
        model.reflect = None
        n01, n10 = int((rr & ~base_rr).sum()), int((~rr & base_rr).sum())
        p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
        res[f"{name}_excluir_RR"] = ex["RR"]
        res[f"{name}_nevir_test"] = nv
        res[f"{name}_vs_own_excluir"] = {"wins": n01, "losses": n10, "p": float(p)}
        per_instance[f"excluir_rr_{name}"] = rr.tolist()
        per_instance[f"nevir_test_{name}"] = nvc.tolist()
        out[name] = (ex["RR"], nv, n01, n10, p)
    print(f"    reflection test (c=+{args.reflect_c:g}, query scope, U from {src}):")
    for name, (rr_, nv, n01, n10, p) in out.items():
        print(f"      {name:<14} ExcluIR RR {rr_:.3f} (own {base_rr.mean():.3f}; "
              f"win/loss {n01}/{n10}, p={p:.2g}) | NevIR test {nv:.3f}")


def make_batches(items, bs, rng, hard):
    items = items[:]
    rng.shuffle(items)
    return [(items[i:i + bs], hard) for i in range(0, len(items) - bs + 1, bs)]


def train_one(args, mode, seed, data, build_model, device):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    model = build_model(mode, seed).to(device)
    if args.grad_checkpointing and mode not in ("zeroshot", "frame_only"):
        model.backbone.gradient_checkpointing_enable()
        model.backbone.config.use_cache = False
    if args.train_top_layers > 0 and mode not in ("zeroshot", "frame_only"):
        n_tr, n_all = freeze_lower_layers(model.backbone, args.train_top_layers)
        print(f"    training top {args.train_top_layers} encoder layers only "
              f"({n_tr / 1e6:.1f}M of {n_all / 1e6:.1f}M encoder parameters)")
    nevir = data["nevir"]
    log = {"mode": mode, "seed": seed, "epochs": [], "U_init": args.U_init,
           "aux_weight": args.aux_weight, "frame_warmup_steps": args.frame_warmup_steps}
    if args.U_init == "probe" and hasattr(model, "U_raw") and model.uses_frame:
        k_model = model.U_raw.shape[1]
        cache = os.path.join(args.out_dir, f"probe_U_k{k_model}.npy")
        U0 = probe_init_U(model, nevir["train"], k_model, args.eval_batch_size, cache)
        model.U_raw.data = torch.as_tensor(U0, dtype=model.U_raw.dtype,
                                           device=model.U_raw.device)
    use_aux = args.aux_weight > 0 and model.uses_frame and mode != "frame_only"

    def evaluate_val():
        acc, _ = eval_nevir(model, nevir["validation"], args.eval_batch_size)
        return acc

    def evaluate_all():
        """Returns (selection score, nevir val, general val).
        General val = nDCG@10 on the out-of-domain BEIR validation split when
        --val_beir is set (sensitive to forgetting), else the in-domain
        held-out MRR@10 (near ceiling; kept only as a fallback)."""
        nv = evaluate_val()
        if data.get("val_beir"):
            gv = eval_ndcg10(model, *data["val_beir"], args.eval_batch_size)
        else:
            gv = eval_general_val(model, data.get("general_val", []), args.eval_batch_size)
        score = nv + gv if (args.select == "combined" and gv == gv) else nv
        return score, nv, gv

    best_score, best_val, gv0 = evaluate_all()
    best_state, best_epoch, best_gv = None, 0, gv0
    log["epochs"].append({"epoch": 0, "val_acc": best_val, "general_val": gv0})
    print(f"    epoch 0  val={best_val:.3f}  general_val={gv0:.3f}")

    if mode != "zeroshot":
        enc_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and n.startswith("backbone.")]
        frame_params = [p for n, p in model.named_parameters()
                        if p.requires_grad and not n.startswith("backbone.")]
        if not model.uses_frame:
            frame_params = []
        groups = []
        if enc_params:
            groups.append({"params": enc_params, "lr": args.lr})
        if frame_params:
            groups.append({"params": frame_params, "lr": args.frame_lr})
        opt = torch.optim.AdamW(groups, weight_decay=0.01)
        best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
        # L2-SP: penalise drift of trainable encoder weights from the
        # pre-trained starting point (anti-forgetting regulariser)
        sp_ref = [(p, p.detach().clone()) for p in enc_params] if args.l2sp > 0 else []
        rng = random.Random(seed)
        trip = nevir_triplets(nevir["train"])
        if args.max_train:
            trip = trip[:args.max_train]
        total_steps = args.epochs * (len(trip) // args.batch_size +
                                     len(data["general"]) // args.batch_size)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / max(1, int(0.1 * total_steps))))

        # Frame warm-up: train U and c alone (encoder frozen) on NevIR triplets,
        # so the frame is already doing work before the encoder can learn
        # around it. Uses a separate optimizer; the encoder schedule is untouched.
        if args.frame_warmup_steps > 0 and model.uses_frame and frame_params \
                and mode != "frame_only":
            for p in enc_params:
                p.requires_grad_(False)
            wopt = torch.optim.AdamW(frame_params, lr=args.frame_lr, weight_decay=0.0)
            wrng = random.Random(seed + 7)
            model.train()
            step, t0 = 0, time.time()
            while step < args.frame_warmup_steps:
                for b, _ in make_batches(trip, args.batch_size, wrng, True):
                    loss = contrastive_loss(model, b, args.temperature, True)
                    if use_aux:
                        loss = loss + args.aux_weight * frame_aux_loss(model, b)
                    wopt.zero_grad()
                    loss.backward()
                    wopt.step()
                    step += 1
                    if step >= args.frame_warmup_steps:
                        break
            for p in enc_params:
                p.requires_grad_(True)
            del wopt
            wval = evaluate_val()
            log["warmup"] = {"steps": step, "val_acc": wval, "c": model.c.item()}
            print(f"    frame warm-up: {step} steps, val={wval:.3f}, "
                  f"c={model.c.item():.3f} ({time.time() - t0:.0f}s)")

        for ep in range(1, args.epochs + 1):
            model.train()
            batches = make_batches(trip, args.batch_size, rng, True) + \
                make_batches(data["general"], args.batch_size, rng, False)
            rng.shuffle(batches)
            t0, run, run_aux = time.time(), 0.0, 0.0
            for bi, (b, hard) in enumerate(batches, 1):
                loss = contrastive_loss(model, b, args.temperature, hard)
                if use_aux and hard:
                    aux = frame_aux_loss(model, b)
                    run_aux += aux.item()
                    loss = loss + args.aux_weight * aux
                if sp_ref:
                    loss = loss + args.l2sp * sum(((p - p0) ** 2).sum() for p, p0 in sp_ref)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at epoch {ep}, step {bi}; stopping "
                                       f"rather than training on NaN (check model precision)")
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                run += loss.item()
                if bi % args.log_every == 0:
                    print(f"      step {bi}/{len(batches)} loss={run / args.log_every:.4f} "
                          + (f"aux={run_aux / args.log_every:.3f} " if use_aux else "")
                          + (f"c={model.c.item():.3f} " if model.uses_frame else "")
                          + f"({time.time() - t0:.0f}s)")
                    run, run_aux = 0.0, 0.0
            score, val, gv = evaluate_all()
            cval = model.c.item() if model.uses_frame else float("nan")
            log["epochs"].append({"epoch": ep, "val_acc": val, "general_val": gv,
                                  "c": cval})
            print(f"    epoch {ep}  val={val:.3f}  general_val={gv:.3f}" +
                  (f"  c={cval:.3f}" if model.uses_frame else ""))
            if score > best_score:
                best_score, best_val, best_gv, best_epoch = score, val, gv, ep
                best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
        model.load_state_dict(best_state)
        del sp_ref

    res = {"mode": mode, "seed": seed, "best_epoch": best_epoch,
           "config": {"U_init": args.U_init, "aux_weight": args.aux_weight,
                      "frame_warmup_steps": args.frame_warmup_steps,
                      "frame_lr": args.frame_lr, "k": args.k, "c_init": args.c_init,
                      "epochs": args.epochs, "batch_size": args.batch_size,
                      "train_top_layers": args.train_top_layers, "l2sp": args.l2sp,
                      "general_n": args.general_n, "lr": args.lr, "select": args.select,
                      "val_beir": args.val_beir},
           "val_acc": best_val, "general_val": best_gv,
           "c": model.c.item() if model.uses_frame else float("nan")}
    res["nevir_test"], corr = eval_nevir(model, nevir["test"], args.eval_batch_size)
    per_instance = {"nevir_test": corr.tolist()}
    if data.get("excluir"):
        rows_x, corpus_x, ids_x = data["excluir"]
        ex, rr = eval_excluir(model, rows_x, args.eval_batch_size, corpus_x, ids_x,
                              full_corpus=args.excluir_full)
        res.update({f"excluir_{k}": v for k, v in ex.items()})
        per_instance["excluir_rr"] = rr.tolist()
    if data.get("scifact"):
        m, qids, sc = eval_ndcg10(model, *data["scifact"], args.eval_batch_size, per_query=True)
        res["scifact_ndcg10"] = m
        # Split by whether the claim contains negation (training lexicon).
        # Hypothesis under test: SciFact counts refuting abstracts as relevant,
        # so polarity-sensitive training should hurt NEGATED claims most.
        qtext = data["scifact"][1]
        neg = np.array([n_cues(qtext[q]) > 0 for q in qids])
        sc = np.array(sc)
        res["scifact_ndcg10_negated"] = float(sc[neg].mean()) if neg.any() else float("nan")
        res["scifact_ndcg10_affirmative"] = float(sc[~neg].mean()) if (~neg).any() else float("nan")
        res["scifact_n_negated"] = int(neg.sum())
        per_instance["scifact_ndcg_q"] = sc.tolist()
        per_instance["scifact_qids"] = list(qids)
        per_instance["scifact_negated"] = neg.tolist()
        print(f"    SciFact nDCG@10 {m:.3f} | negated claims ({neg.sum()}) "
              f"{res['scifact_ndcg10_negated']:.3f} | affirmative ({(~neg).sum()}) "
              f"{res['scifact_ndcg10_affirmative']:.3f}")
    if args.reflect_eval and data.get("excluir"):
        reflect_eval(args, model, mode, data, res, per_instance)
    if args.content_eval:
        content_eval(args, model, mode, data, res, per_instance)
    if args.analysis_pack and mode in ("zeroshot", "plain_ft"):
        import so_analysis
        so_analysis.analysis_pack(args, model, mode, data, res, per_instance)
    res["log"] = log
    if mode != "zeroshot":
        del opt, sched, best_state
    model.to("cpu")                      # move weights off the GPU before release
    del model
    return res, per_instance

# ---------------------------------------------------------------------------
# Smoke test: tiny random BERT + word-level tokenizer, synthetic data
# ---------------------------------------------------------------------------


def smoke_setup():
    from tokenizers import Tokenizer, models, pre_tokenizers, processors, trainers
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast
    nev = synthetic_nevir(n_train=160, n_val=60, n_test=60)
    corpus = [t for rows in nev.values() for r in rows for t in r]
    tk = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.train_from_iterator(corpus, trainers.WordLevelTrainer(
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"]))
    tk.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", tk.token_to_id("[CLS]")), ("[SEP]", tk.token_to_id("[SEP]"))])
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]",
                                  cls_token="[CLS]", sep_token="[SEP]")
    cfg = BertConfig(vocab_size=tk.get_vocab_size(), hidden_size=32, num_hidden_layers=2,
                     num_attention_heads=2, intermediate_size=64, max_position_embeddings=64)
    exc = [(r[1], r[3], r[2]) for r in nev["test"]]       # (negated q, d+, d-) stand-in
    docs = {f"d{i}": r[2] for i, r in enumerate(nev["test"][:40])}
    qs = {f"q{i}": r[0] for i, r in enumerate(nev["test"][:40])}
    rel = {f"q{i}": {f"d{i}"} for i in range(40)}
    return tok, cfg, nev, exc, (docs, qs, rel)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def summarise(all_res, out_dir):
    keys = [k for k in ("val_acc", "general_val", "nevir_test", "excluir_RR", "excluir_dR@1",
                        "excluir_dMRR@10", "excluir_off_R@1", "excluir_off_dR@1",
                        "excluir_off_dMRR@10", "excluir_off_RR", "scifact_ndcg10",
                        "scifact_ndcg10_negated", "scifact_ndcg10_affirmative",
                        "content_scifact_ndcg10", "content_randU_scifact_ndcg10",
                        "content_nevir_test", "content_excluir_RR",
                        "reflect_excluir_RR", "reflect_randU_excluir_RR",
                        "reflect_nevir_test", "c")
            if any(k in r for r in all_res)]
    modes = list(dict.fromkeys(r["mode"] for r in all_res))
    rows = []
    print("\n=== Summary (mean ± sd over seeds) ===")
    print("  " + f"{'mode':<15}" + "".join(f"{k:>18}" for k in keys))
    for m in modes:
        rs = [r for r in all_res if r["mode"] == m]
        row = {"mode": m, "n_seeds": len(rs)}
        line = f"  {m:<15}"
        for k in keys:
            v = np.array([r.get(k, float("nan")) for r in rs], dtype=float)
            row[k + "_mean"], row[k + "_sd"] = float(v.mean()), float(v.std(ddof=0))
            line += f"{v.mean():>11.3f} ± {v.std(ddof=0):.3f}"
        rows.append(row)
        print(line)
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def paired_tests(per_inst, out_dir, ref="plain_ft"):
    """Exact McNemar per seed: each mode vs the reference fine-tuning."""
    from scipy.stats import binomtest
    lines = []
    for (mode, seed), pi in per_inst.items():
        if mode == ref or (ref, seed) not in per_inst:
            continue
        for metric in pi:
            if (metric not in per_inst[(ref, seed)] or metric.startswith("scifact_")
                    or metric.endswith(("_sci_q", "_ctrl_q")) or metric == "nevir_explicit"):
                continue                       # McNemar applies to binary outcomes only
            a = np.array(per_inst[(ref, seed)][metric], dtype=bool)
            b = np.array(pi[metric], dtype=bool)
            n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
            p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
            lines.append({"mode": mode, "seed": seed, "metric": metric,
                          f"wins_vs_{ref}": n01, f"losses_vs_{ref}": n10, "p": float(p)})
    if lines:
        print(f"\n=== Paired tests vs {ref} (exact McNemar, per seed) ===")
        for l in lines:
            print(f"  {l['mode']:<15} seed {l['seed']}  {l['metric']:<12} "
                  f"win/loss {l[f'wins_vs_{ref}']}/{l[f'losses_vs_{ref}']}  p={l['p']:.3g}")
        with open(os.path.join(out_dir, "paired_tests.json"), "w") as f:
            json.dump(lines, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="intfloat/e5-base-v2")
    ap.add_argument("--modes", nargs="+",
                    default=["zeroshot", "plain_ft", "frame", "frame_randomU"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--c_init", type=float, default=-2.0,
                    help="initial frame gain; -2 = amplification selected by the probe")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--eval_batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--frame_lr", type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--max_train", type=int, default=0,
                    help="cap on NevIR training triplets (0 = all)")
    ap.add_argument("--general_hf", default="sentence-transformers/natural-questions")
    ap.add_argument("--general_n", type=int, default=4000,
                    help="ordinary retrieval pairs mixed in (0 disables)")
    ap.add_argument("--excluir", default=None,
                    help="ExcluIR test file, e.g. test_manual_final.json")
    ap.add_argument("--excluir_corpus", default=None,
                    help="corpus.json (default: same folder as --excluir)")
    ap.add_argument("--excluir_full", action="store_true",
                    help="also run the authors' full-corpus protocol (90k docs; "
                         "re-encoded for every model, so GPU recommended)")
    ap.add_argument("--scifact", action="store_true", help="evaluate BEIR SciFact nDCG@10")
    ap.add_argument("--device", default=None)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--out_dir", default="so_train_results")
    ap.add_argument("--smoke_test", action="store_true")
    ap.add_argument("--U_init", choices=["random", "probe"], default="random",
                    help="probe = start U from the frozen encoder's fitted negation "
                         "subspace (NevIR train pairs); frame_randomU is unaffected")
    ap.add_argument("--frame_warmup_steps", type=int, default=0,
                    help="train only U and c for this many steps before the encoder")
    ap.add_argument("--aux_weight", type=float, default=0.0,
                    help="weight of the frame auxiliary loss (negation difference "
                         "must lie in span(U)); applied in all frame modes, "
                         "including frame_randomU, so the control stays matched")
    ap.add_argument("--train_top_layers", type=int, default=0,
                    help="anti-forgetting: train only the top N encoder layers "
                         "(0 = all layers, the original setting)")
    ap.add_argument("--l2sp", type=float, default=0.0,
                    help="anti-forgetting: weight of the L2-SP penalty pulling encoder "
                         "weights towards their pre-trained values (0 = off)")
    ap.add_argument("--general_val_n", type=int, default=1000,
                    help="held-out general-retrieval pairs for validation (disjoint "
                         "from the training pairs); reports MRR@10 every epoch")
    ap.add_argument("--val_beir", default="",
                    help="out-of-domain BEIR dataset whose VALIDATION split measures "
                         "general retrieval during training, e.g. nfcorpus (never the "
                         "test benchmark itself)")
    ap.add_argument("--select", choices=["nevir", "combined"], default="nevir",
                    help="checkpoint selection: NevIR val accuracy (original), or "
                         "NevIR val + general val MRR@10 (guards against forgetting)")
    ap.add_argument("--reflect_eval", action="store_true",
                    help="pre-registered reflection test after training: query-scope "
                         "reflection (c=+2) on ExcluIR and NevIR, plus random-U control")
    ap.add_argument("--analysis_pack", action="store_true",
                    help="revision analysis (so_analysis.py) after zero-shot / plain "
                         "fine-tuned models: rank sweep, baselines, stability, held-out "
                         "fit, controlled negation, BEIR preservation")
    ap.add_argument("--analysis_beir", nargs="*", default=["nfcorpus", "fiqa"],
                    help="BEIR test sets for the preservation check")
    ap.add_argument("--ctrl_file", default=None,
                    help="JSON of verified SciFact claim variants for the controlled-negation "
                         "test: [{qid, variants: {name: text}}]")
    ap.add_argument("--analysis_light", action="store_true",
                    help="evaluate only full, U8, U8_plus_d, d_only and rand8 readouts (faster)")
    ap.add_argument("--content_eval", action="store_true",
                    help="pre-registered content-readout test: project the frame "
                         "subspace out of queries and documents (polarity-blind "
                         "'aboutness' score), plus a random-U control")
    ap.add_argument("--reflect_c", type=float, default=2.0,
                    help="reflection gain; +2 is the pre-registered value, do not tune")
    ap.add_argument("--tag", default="",
                    help="suffix for result labels, e.g. _noAux for ablation variants")
    ap.add_argument("--grad_checkpointing", action="store_true",
                    help="recompute encoder activations in the backward pass: much "
                         "less GPU memory for ~30%% more compute, same results")
    ap.add_argument("--no_exclusion_cues", action="store_true",
                    help="use the probe's original negation-only cue lexicon")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    use_exclusion_cues(not args.no_exclusion_cues)

    device = args.device or ("cuda" if torch.cuda.is_available() else
                             "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    if args.smoke_test:
        from transformers import BertModel
        tok, cfg, nev, exc, sci = smoke_setup()
        data = {"nevir": nev, "general": [(r[0], r[2]) for r in nev["train"][:64]],
                "general_val": [(r[0], r[2]) for r in nev["validation"][:40]],
                "excluir": (exc, [r[1] for r in exc] + [r[2] for r in exc],
                            [(i, len(exc) + i) for i in range(len(exc))]),
                "scifact": sci}
        if args.val_beir:
            data["val_beir"] = sci          # stand-in for the BEIR validation split
        if args.analysis_pack:
            data["analysis_beir"] = {"nfcorpus": sci}
        args.excluir_full = True
        if "--modes" not in sys.argv:
            args.modes = ["zeroshot", "plain_ft", "frame", "frame_randomU",
                          "global_frame", "frame_only"]
        args.epochs, args.batch_size, args.log_every = 2, 16, 10
        args.lr = args.frame_lr = 1e-3
        e5 = False

        def build_model(mode, seed):
            torch.manual_seed(1000)                 # identical backbone init across modes
            return FrameEncoder(BertModel(cfg), tok, mode, k=2, c_init=args.c_init,
                                max_len=64, e5_prefix=e5, seed=seed)
    else:
        from sentence_transformers import SentenceTransformer
        print("Loading data ...")
        data = {"nevir": load_nevir()}
        data["general"], data["general_val"] = load_general(
            args.general_hf, args.general_n, 0, args.general_val_n)
        if args.excluir:
            data["excluir"] = load_excluir(args.excluir, args.excluir_corpus)
            cb, ce = scope_coverage([r[0] for r in data["excluir"][0]])
            print(f"  ExcluIR queries with a detected scope: training detector {cb:.1%}, "
                  f"extended (reflection-test) detector {ce:.1%}")
        if args.scifact:
            data["scifact"] = load_scifact()
        if args.val_beir:
            data["val_beir"] = load_beir(args.val_beir, ("validation", "dev"))
        if args.analysis_pack:
            if "scifact" not in data:
                data["scifact"] = load_scifact()
            data["analysis_beir"] = {n: load_beir(n, ("test",)) for n in args.analysis_beir}
        print("  NevIR " + ", ".join(f"{k}={len(v)}" for k, v in data["nevir"].items())
              + f"; general pairs={len(data['general'])} train, "
              f"{len(data['general_val'])} held-out validation")
        e5 = "e5" in args.model.lower()

        def build_model(mode, seed):
            st = SentenceTransformer(args.model, device="cpu")
            backbone, tok = st[0].auto_model, st.tokenizer
            # Some checkpoints (e.g. thenlper/gte-base) ship 16-bit weights;
            # training them directly with AdamW overflows to NaN at once.
            if next(backbone.parameters()).dtype != torch.float32:
                backbone = backbone.float()
            del st
            return FrameEncoder(backbone, tok, mode, k=args.k, c_init=args.c_init,
                                max_len=args.max_len, e5_prefix=e5, seed=seed)

    # Results accumulate across separate invocations with the same --out_dir,
    # so each mode can run in its own process (guarantees a clean GPU on MPS).
    # A re-run of the same (mode, seed) replaces the earlier entry.
    res_path = os.path.join(args.out_dir, "results.json")
    pi_path = os.path.join(args.out_dir, "per_instance.json")
    all_res, per_inst = [], {}
    if os.path.exists(res_path):
        with open(res_path) as f:
            all_res = json.load(f)
        if os.path.exists(pi_path):
            with open(pi_path) as f:
                per_inst = {(d["mode"], d["seed"]): d["data"] for d in json.load(f)}
        print(f"  continuing {res_path}: {len(all_res)} earlier result(s) "
              f"({', '.join(sorted({r['mode'] for r in all_res}))})")
    for seed in args.seeds:
        for mode in args.modes:
            if mode == "zeroshot" and seed != args.seeds[0]:
                continue                               # deterministic; run once
            label = mode + args.tag
            print(f"\n--- mode={label} seed={seed} ---")
            t0 = time.time()
            res, pi = train_one(args, mode, seed, data, build_model, device)
            res["minutes"] = (time.time() - t0) / 60
            res["mode"], res["base_mode"] = label, mode
            all_res = [r for r in all_res if (r["mode"], r["seed"]) != (label, seed)]
            all_res.append(res)
            per_inst[(label, seed)] = pi
            with open(pi_path, "w") as f:
                json.dump([{"mode": m, "seed": s, "data": d}
                           for (m, s), d in per_inst.items()], f)
            print(f"    -> NevIR test {res['nevir_test']:.3f}"
                  + (f" | ExcluIR RR {res['excluir_RR']:.3f}" if "excluir_RR" in res else "")
                  + (f" | SciFact nDCG@10 {res['scifact_ndcg10']:.3f}" if "scifact_ndcg10" in res else "")
                  + f"  ({res['minutes']:.1f} min)")
            with open(res_path, "w") as f:
                json.dump(all_res, f, indent=2)
            import gc
            gc.collect()                 # break optimizer/scheduler reference cycles
            if device == "mps":
                torch.mps.empty_cache()
            elif device == "cuda":
                torch.cuda.empty_cache()

    summarise(all_res, args.out_dir)
    paired_tests(per_inst, args.out_dir)
    print(f"\nResults in {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
