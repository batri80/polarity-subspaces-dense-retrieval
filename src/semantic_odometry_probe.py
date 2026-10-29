#!/usr/bin/env python3
"""
Semantic Odometry -- feasibility probe (v0.1)
=============================================

Tests the central empirical claim behind the Semantic Odometry paradigm:
negation behaves like a *frame change* (a scoped reflection on a low-rank
"frame" subspace) rather than as added content, and transporting scoped
tokens back into a canonical frame before pooling improves negation-sensitive
retrieval with a plain bi-encoder.

Two probes are run for every encoder:

  Probe A (geometry). On affirmative/negated pairs, fit five candidate maps
  from the affirmative embedding to the negated embedding on a training split
  and score them on held-out pairs:
      identity            b = a
      translation         b = a + d
      global rotation     b = W a,  W orthogonal (Procrustes; RISE-style)
      subspace frame      b = a - c U U^T a,  U in R^{d x k} orthonormal
                          (c = 1: projection removal, Widdows-style;
                           c = 2: Householder reflection;  c free: fitted)
      ridge linear        b = M a   (unconstrained upper bound)

  Probe B (retrieval, NevIR). Pairwise accuracy of a bi-encoder under:
      plain               ordinary mean pooling
      global frame        whole embedding transported if the text carries an
                          odd number of negation cues (static, RISE-like)
      scoped frame        only tokens inside negation scope are transported
                          before pooling; nested cues compose by parity, so
                          double negation cancels (R^2 = I) by construction
      scoped, random U    same as scoped but with a random subspace (control)
  k and c are selected on the NevIR validation split and reported on test,
  with bootstrap 95% CIs and an exact McNemar test against plain pooling.

  READ THE CONTROL FIRST. A negated query and its negated document share the
  same scoped words, so transporting scoped tokens by *any* transform can make
  them match ("scope marking"). The random-U control is drawn afresh at every
  grid point and also tuned on validation, so it is a deliberately strong
  adversary. The paradigm's geometric claim is supported only where the
  learned frame beats random U; a gain shared by both supports the weaker
  claim that explicit scope separation helps a bi-encoder.

  The frame fit is offset-aware (b = a + d - c U U^T a): the cue word's own
  vector appears as a near-constant translation d, and fitting U without d
  lets that direction swallow the frame subspace (verified on the synthetic
  encoder: ~70 deg error without the offset, ~10 deg with it).

Usage
-----
  pip install -U sentence-transformers datasets scipy numpy
  python semantic_odometry_probe.py --synthetic            # self-test, no downloads
  python semantic_odometry_probe.py                        # default encoders on NevIR
  python semantic_odometry_probe.py --models sentence-transformers/all-mpnet-base-v2
  python semantic_odometry_probe.py --pairs_csv health_pairs.csv   # extra Probe A on
                                                           # your own aff,neg pairs

Notes
-----
* Token-level transport is exact only for mean-pooled encoders. For a CLS-pooled
  model (e.g. BGE) the script mean-pools the token states itself for *every*
  condition, so comparisons stay internally fair, and prints a warning.
* The negation scope detector is a deliberately simple, transparent heuristic
  (cue -> next clause boundary). Swapping in the NCSR cue/scope model is the
  obvious upgrade once the probe confirms the direction.
"""

import argparse
import csv
import json
import os
import re
import sys
import time

import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.stats import binomtest

DEFAULT_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "intfloat/e5-base-v2",
]

# ---------------------------------------------------------------------------
# Negation cue and scope detection (character spans -> per-word parity)
# ---------------------------------------------------------------------------

CUES = {
    "not", "no", "never", "none", "nobody", "nothing", "nowhere", "neither",
    "nor", "cannot", "without", "lack", "lacks", "lacking", "absent", "absence",
}
BOUNDARIES = {
    ".", ",", ";", ":", "!", "?", "but", "however", "although", "though",
    "yet", "whereas", "while", "except",
}
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]")


def is_cue(w):
    w = w.lower()
    return w in CUES or w.endswith("n't")


def word_parities(text):
    """Return list of (start, end, parity, is_cue) for each word.

    parity = number of cues seen since the last clause boundary, mod 2.
    Cue words themselves keep the parity in force *before* them, so the cue
    token is not transported by its own frame. This makes nested negation
    compose as a product of reflections, and an even count cancels.
    """
    out, count = [], 0
    for m in WORD_RE.finditer(text):
        w = m.group(0)
        lw = w.lower()
        if lw in BOUNDARIES:
            count = 0
            out.append((m.start(), m.end(), 0, False))
        elif is_cue(w):
            out.append((m.start(), m.end(), count % 2, True))
            count += 1
        else:
            out.append((m.start(), m.end(), count % 2, False))
    return out


def n_cues(text):
    return sum(1 for m in WORD_RE.finditer(text) if is_cue(m.group(0)))


def token_parities(text, offsets):
    """Map tokenizer offsets onto word-level parities (0 for special tokens)."""
    words = word_parities(text)
    par = np.zeros(len(offsets), dtype=np.int8)
    wi = 0
    for ti, (s, e) in enumerate(offsets):
        if e <= s:                       # special / padding token
            continue
        while wi < len(words) and words[wi][1] <= s:
            wi += 1
        if wi < len(words) and words[wi][0] <= s < words[wi][1]:
            par[ti] = words[wi][2]
    return par

# ---------------------------------------------------------------------------
# Encoders. Each returns, per text:
#   h   mean-pooled token states (un-normalised)
#   s   (sum of token states with odd parity) / n_tokens
#   odd whether the text carries an odd number of cues (for the global frame)
# so every condition can be computed cheaply afterwards:
#   plain   h
#   global  h - c P h          if odd
#   scoped  h - c P s
# ---------------------------------------------------------------------------


class STEncoder:
    def __init__(self, name, device=None, batch_size=64):
        from sentence_transformers import SentenceTransformer
        self.name = name
        self.model = SentenceTransformer(name, device=device)
        self.tok = self.model.tokenizer
        self.batch_size = batch_size
        self.max_len = self.model.max_seq_length
        pool = None
        for mod in self.model:
            if mod.__class__.__name__ == "Pooling":
                pool = mod
        if pool is not None and not getattr(pool, "pooling_mode_mean_tokens", False):
            print(f"  [warn] {name} does not use mean pooling; mean-pooling token "
                  f"states for all conditions so comparisons stay fair.")
        self.is_e5 = "e5" in name.lower()

    def prefix(self, text, role):
        if not self.is_e5:
            return text
        return ("query: " if role == "query" else "passage: ") + text

    def encode(self, texts):
        tok_embs = self.model.encode(
            texts, output_value="token_embeddings", batch_size=self.batch_size,
            convert_to_numpy=False, show_progress_bar=True)
        H, S, ODD = [], [], []
        for text, te in zip(texts, tok_embs):
            te = te.detach().float().cpu().numpy()
            enc = self.tok(text, truncation=True, max_length=self.max_len,
                           return_offsets_mapping=True)
            offs = enc["offset_mapping"]
            n = min(len(offs), te.shape[0])
            te, offs = te[:n], offs[:n]
            par = token_parities(text, offs)
            H.append(te.mean(0))
            S.append(te[par == 1].sum(0) / n if par.any() else np.zeros(te.shape[1]))
            ODD.append(n_cues(text) % 2 == 1)
        return np.array(H), np.array(S), np.array(ODD)


class SyntheticEncoder:
    """Toy encoder with a planted frame subspace, for testing the pipeline.

    Word vectors are random. Inside negation scope the encoder applies a
    *weak* partial reflection on a planted rank-2 subspace (mimicking real
    encoders, which register negation only faintly), and the cue word adds a
    small vector. Probe A should recover the planted subspace.
    """

    def __init__(self, dim=64, rank=2, strength=0.3, cue_scale=0.05,
                 noise=0.005, seed=0):
        self.rng = np.random.default_rng(seed)
        self.dim, self.strength = dim, strength
        self.cue_scale, self.noise = cue_scale, noise
        q, _ = np.linalg.qr(self.rng.standard_normal((dim, rank)))
        self.U_true = q
        self.vocab = {}
        self.name = "synthetic"

    def prefix(self, text, role):
        return text

    def vec(self, w):
        w = w.lower()
        if w not in self.vocab:
            self.vocab[w] = self.rng.standard_normal(self.dim)
        return self.vocab[w]

    def encode(self, texts):
        P = self.U_true @ self.U_true.T
        H, S, ODD = [], [], []
        for text in texts:
            toks = []
            for (s, e, par, cue) in word_parities(text):
                v = self.vec(text[s:e]) * (self.cue_scale if cue else 1.0)
                if par == 1:
                    v = v - self.strength * (P @ v)
                toks.append((v, par))
            te = np.array([t[0] for t in toks])
            par = np.array([t[1] for t in toks])
            n = len(te)
            H.append(te.mean(0) + self.noise * self.rng.standard_normal(self.dim))
            S.append(te[par == 1].sum(0) / n if par.any() else np.zeros(self.dim))
            ODD.append(n_cues(text) % 2 == 1)
        return np.array(H), np.array(S), np.array(ODD)

# ---------------------------------------------------------------------------
# Probe A: candidate maps aff -> neg
# ---------------------------------------------------------------------------


def fit_frame(A, B, k, c=None, offset=True, rounds=4):
    """Fit b = a + d - c * a U U^T  (offset-aware frame model).

    The cue word contributes its own vector, which appears as a near-constant
    translation d. Fitting U without d lets that cue-content direction swallow
    the frame subspace, so d and (U, c) are estimated by alternation.
    Returns U, c, d (d is zero when offset=False).
    """
    d = (B - A).mean(0) if offset else np.zeros(A.shape[1])
    for _ in range(rounds if offset else 1):
        U, cc = fit_subspace(A, B - d, k, c=c)
        if offset:
            d = (B - A + cc * (A @ U) @ U.T).mean(0)
    return U, cc, d


def fit_subspace(A, B, k, c=None, iters=300, seed=0):
    """Fit b = a - c * a U U^T with U (d x k) orthonormal.

    If c is None it is fitted jointly (closed form given U).
    Riemannian gradient descent on the Stiefel manifold with QR retraction
    and backtracking; initialised from the top-k right singular vectors of
    M = A - B, which is exact in the noiseless case.
    """
    M = A - B
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    U = Vt[:k].T.copy()

    def c_star(U):
        AP = (A @ U) @ U.T
        den = float((AP * AP).sum())
        return float((M * AP).sum()) / den if den > 0 else 0.0

    def loss(U, cc):
        R = M - cc * (A @ U) @ U.T
        return float((R * R).sum()), R

    cc = c_star(U) if c is None else c
    L, R = loss(U, cc)
    lr = 1.0 / (np.linalg.norm(A, 2) ** 2 + 1e-9)
    for _ in range(iters):
        G = -2.0 * cc * (A.T @ (R @ U) + R.T @ (A @ U))
        G = G - U @ ((U.T @ G + G.T @ U) / 2.0)          # tangent projection
        step = lr
        improved = False
        for _ in range(20):
            Un, _ = np.linalg.qr(U - step * G)
            cn = c_star(Un) if c is None else c
            Ln, Rn = loss(Un, cn)
            if Ln < L:
                U, cc, L, R = Un, cn, Ln, Rn
                lr = step * 1.5
                improved = True
                break
            step *= 0.5
        if not improved:
            break
    return U, cc


def l2n(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def geometry_metrics(pred, aff, neg):
    P, A, N = l2n(pred), l2n(aff), l2n(neg)
    cos_true = (P * N).sum(1)
    cos_aff = (P * A).sum(1)
    sims = P @ np.vstack([N, A]).T                 # candidates: all negs + affs
    top1 = (sims.argmax(1) == np.arange(len(P))).mean()
    # Fidelity: 1 - ||pred - neg|| / ||aff - neg|| on unit vectors. Positive only
    # if the map moves the prediction TOWARDS the true negation relative to the
    # identity map; pushing it away from aff in a wrong direction scores < 0.
    # "closer" and "top1" alone can be gamed by large displacements.
    err_red = 1.0 - np.linalg.norm(P - N, axis=1) / (np.linalg.norm(A - N, axis=1) + 1e-12)
    return {
        "cos_to_true_neg": float(cos_true.mean()),
        "closer_to_neg_than_aff": float((cos_true > cos_aff).mean()),
        "top1_retrieval": float(top1),
        "err_reduction": float(np.median(err_red)),
        "margin": float((cos_true - cos_aff).mean()),
    }


def probe_geometry(Atr, Btr, Ate, Bte, ranks, label):
    rows = []

    def add(name, pred, extra=""):
        m = geometry_metrics(pred, Ate, Bte)
        m.update({"pairs": label, "map": name, "params": extra})
        rows.append(m)

    add("identity", Ate)
    d = (Btr - Atr).mean(0)
    add("translation", Ate + d)
    W, _ = orthogonal_procrustes(Atr, Btr)
    add("global rotation (Procrustes)", Ate @ W)
    for k in ranks:
        for cfix, nm in [(1.0, "projection c=1"), (2.0, "reflection c=2"), (None, "fitted c")]:
            U, cc = fit_subspace(Atr, Btr, k, c=cfix)
            add(f"subspace {nm}", Ate - cc * (Ate @ U) @ U.T, f"k={k}, c={cc:.3f}")
        U, cc, d = fit_frame(Atr, Btr, k, c=None)
        add("cue offset + frame (fitted c)", Ate + d - cc * (Ate @ U) @ U.T,
            f"k={k}, c={cc:.3f}")
    lam = 1.0
    Mr = np.linalg.solve(Atr.T @ Atr + lam * np.eye(Atr.shape[1]), Atr.T @ Btr)
    add("ridge linear (upper bound)", Ate @ Mr, f"lambda={lam}")
    return rows

# ---------------------------------------------------------------------------
# Probe B: NevIR pairwise accuracy
# ---------------------------------------------------------------------------


def transport(H, S, ODD, U, c, mode):
    if mode == "plain" or U is None:
        return H
    if mode == "global":
        Hp = H - c * (H @ U) @ U.T
        return np.where(ODD[:, None], Hp, H)
    if mode == "scoped":
        return H - c * (S @ U) @ U.T
    raise ValueError(mode)


def nevir_correct(emb, idx):
    """Per-instance pairwise correctness: q1 prefers d1 AND q2 prefers d2."""
    E = l2n(emb)
    q1, q2, d1, d2 = (E[idx[:, j]] for j in range(4))
    s = lambda x, y: (x * y).sum(1)
    return (s(q1, d1) > s(q1, d2)) & (s(q2, d2) > s(q2, d1))


def bootstrap_ci(x, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    stats = [x[rng.integers(0, n, n)].mean() for _ in range(B)]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def mcnemar(a, b):
    n01 = int((~a & b).sum())
    n10 = int((a & ~b).sum())
    if n01 + n10 == 0:
        return 1.0, n01, n10
    return float(binomtest(n01, n01 + n10, 0.5).pvalue), n01, n10

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_nevir(csv_dir=None):
    """Return dict split -> list of (q1, q2, doc1, doc2)."""
    splits = {}
    if csv_dir:
        for sp in ("train", "validation", "test"):
            p = os.path.join(csv_dir, f"{sp}.csv")
            with open(p, newline="", encoding="utf-8") as f:
                splits[sp] = [(r["q1"], r["q2"], r["doc1"], r["doc2"])
                              for r in csv.DictReader(f)]
        return splits
    from datasets import load_dataset
    ds = load_dataset("orionweller/NevIR")
    for sp in ("train", "validation", "test"):
        splits[sp] = [(r["q1"], r["q2"], r["doc1"], r["doc2"]) for r in ds[sp]]
    return splits


def synthetic_nevir(n_train=600, n_val=200, n_test=300, seed=1):
    rng = np.random.default_rng(seed)
    nouns = [f"w{i}" for i in range(400)]

    def inst():
        subj = list(rng.choice(nouns, 3, replace=False))
        pred = list(rng.choice(nouns, 4, replace=False))
        d1 = " ".join(subj + ["is"] + pred) + "."
        d2 = " ".join(subj + ["is", "not"] + pred) + "."
        q1 = " ".join(subj[:2] + ["is"] + pred[:2]) + " ?"
        q2 = " ".join(subj[:2] + ["is", "not"] + pred[:2]) + " ?"
        return (q1, q2, d1, d2)

    return {"train": [inst() for _ in range(n_train)],
            "validation": [inst() for _ in range(n_val)],
            "test": [inst() for _ in range(n_test)]}


def orient_pairs(pairs):
    """Turn (x, y) text pairs into (affirmative, negated) by cue count."""
    out = []
    for x, y in pairs:
        cx, cy = n_cues(x), n_cues(y)
        if cy > cx:
            out.append((x, y))
        elif cx > cy:
            out.append((y, x))
    return out


def load_pairs_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [(r["aff"], r["neg"]) for r in csv.DictReader(f)]

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_encoder(enc, nevir, pairs_extra, ranks, out_dir, cache_dir):
    tag = enc.name.replace("/", "__")
    os.makedirs(out_dir, exist_ok=True)

    # 1. collect every text with its role, encode once (cached)
    roles = {}
    for sp, rows in nevir.items():
        for q1, q2, d1, d2 in rows:
            roles[enc.prefix(q1, "query")] = 1
            roles[enc.prefix(q2, "query")] = 1
            roles[enc.prefix(d1, "doc")] = 1
            roles[enc.prefix(d2, "doc")] = 1
    for a, b in pairs_extra:
        roles[enc.prefix(a, "doc")] = 1
        roles[enc.prefix(b, "doc")] = 1
    texts = list(roles)
    cache = os.path.join(cache_dir, f"{tag}.npz") if cache_dir else None
    H = None
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        if list(z["texts"]) == texts:
            H, S, ODD = z["H"], z["S"], z["ODD"]
    if H is None:
        t0 = time.time()
        H, S, ODD = enc.encode(texts)
        print(f"  encoded {len(texts)} texts in {time.time()-t0:.1f}s")
        if cache:
            os.makedirs(cache_dir, exist_ok=True)
            np.savez(cache, texts=np.array(texts, dtype=object), H=H, S=S, ODD=ODD)
    pos = {t: i for i, t in enumerate(texts)}
    ix = lambda t, role: pos[enc.prefix(t, role)]

    # 2. Probe A on NevIR doc and query pairs (train -> fit, test -> score)
    def pair_idx(split):
        prs = []
        for q1, q2, d1, d2 in nevir[split]:
            for a, b in orient_pairs([(d1, d2)]):
                prs.append((ix(a, "doc"), ix(b, "doc")))
            for a, b in orient_pairs([(q1, q2)]):
                prs.append((ix(a, "query"), ix(b, "query")))
        return np.array(prs)

    tr, te = pair_idx("train"), pair_idx("test")
    geo = probe_geometry(H[tr[:, 0]], H[tr[:, 1]], H[te[:, 0]], H[te[:, 1]],
                         ranks, "NevIR")
    if pairs_extra:
        rng = np.random.default_rng(0)
        pi = np.array([(ix(a, "doc"), ix(b, "doc")) for a, b in pairs_extra])
        perm = rng.permutation(len(pi))
        cut = int(0.7 * len(pi))
        a_tr, a_te = pi[perm[:cut]], pi[perm[cut:]]
        geo += probe_geometry(H[a_tr[:, 0]], H[a_tr[:, 1]], H[a_te[:, 0]],
                              H[a_te[:, 1]], ranks, "user pairs")

    # 3. Probe B: fit U on train pairs, select (k, c) on validation, test once
    def inst_idx(split):
        return np.array([[ix(q1, "query"), ix(q2, "query"), ix(d1, "doc"), ix(d2, "doc")]
                         for q1, q2, d1, d2 in nevir[split]])

    vi, ti = inst_idx("validation"), inst_idx("test")
    fits = {}
    for k in ranks:
        for cfix in (1.0, 2.0, None):
            U, cc, _ = fit_frame(H[tr[:, 0]], H[tr[:, 1]], k, c=cfix)
            fits[(k, "fit" if cfix is None else cfix)] = (U, cc)

    rng = np.random.default_rng(123)
    results, correct = [], {}
    plain_te = nevir_correct(H, ti)
    correct["plain"] = plain_te
    for mode in ("global", "scoped", "scoped_randomU"):
        best = None
        for (k, ckey), (U, cc) in fits.items():
            for csign in (cc, -cc):           # amplify or flip along the axis
                if mode == "scoped_randomU":
                    Ur, _ = np.linalg.qr(rng.standard_normal(U.shape))
                    Uuse, m = Ur, "scoped"
                else:
                    Uuse, m = U, mode
                acc_v = nevir_correct(transport(H, S, ODD, Uuse, csign, m), vi).mean()
                if best is None or acc_v > best[0]:
                    best = (acc_v, k, csign, Uuse, m)
        acc_v, k, csign, Uuse, m = best
        corr = nevir_correct(transport(H, S, ODD, Uuse, csign, m), ti)
        correct[mode] = corr
        results.append((mode, k, csign, acc_v, corr))

    table = []
    lo, hi = bootstrap_ci(plain_te.astype(float))
    table.append({"condition": "plain", "k": "", "c": "", "val_acc": "",
                  "test_acc": float(plain_te.mean()), "ci95": [lo, hi],
                  "mcnemar_p_vs_plain": "", "wins/losses": ""})
    for mode, k, csign, acc_v, corr in results:
        lo, hi = bootstrap_ci(corr.astype(float))
        p, n01, n10 = mcnemar(plain_te, corr)
        table.append({"condition": mode, "k": k, "c": round(csign, 3),
                      "val_acc": float(acc_v), "test_acc": float(corr.mean()),
                      "ci95": [lo, hi], "mcnemar_p_vs_plain": p,
                      "wins/losses": f"{n01}/{n10}"})

    # 4. write outputs
    with open(os.path.join(out_dir, f"{tag}_geometry.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(geo[0].keys()))
        w.writeheader()
        w.writerows(geo)
    with open(os.path.join(out_dir, f"{tag}_nevir.json"), "w") as f:
        json.dump(table, f, indent=2)
    return geo, table


def print_report(name, geo, table):
    print(f"\n=== {name} ===")
    print("Probe A  (held-out aff->neg maps; higher is better)")
    print(f"  {'pairs':<11}{'map':<34}{'params':<18}{'cos':>7}{'closer':>8}{'top1':>7}"
          f"{'err_red':>9}{'margin':>9}")
    for r in geo:
        print(f"  {r['pairs']:<11}{r['map']:<34}{r['params']:<18}"
              f"{r['cos_to_true_neg']:7.3f}{r['closer_to_neg_than_aff']:8.3f}"
              f"{r['top1_retrieval']:7.3f}{r['err_reduction']:9.3f}{r['margin']:9.4f}")
    print("  err_red = median fractional error reduction vs identity (>0 means the map")
    print("  moves predictions towards the true negation; robust to distortion)")
    print("Probe B  (NevIR pairwise accuracy, test split; random ranking = 0.25)")
    print(f"  {'condition':<16}{'k':>4}{'c':>8}{'val':>8}{'test':>8}   {'95% CI':<16}"
          f"{'p vs plain':>11}{'win/loss':>10}")
    for r in table:
        ci = f"[{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}]"
        val = f"{r['val_acc']:.3f}" if r["val_acc"] != "" else ""
        p = f"{r['mcnemar_p_vs_plain']:.2g}" if r["mcnemar_p_vs_plain"] != "" else ""
        print(f"  {r['condition']:<16}{str(r['k']):>4}{str(r['c']):>8}{val:>8}"
              f"{r['test_acc']:8.3f}   {ci:<16}{p:>11}{r['wins/losses']:>10}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--ranks", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--nevir_csv_dir", default=None,
                    help="folder with train/validation/test.csv (q1,q2,doc1,doc2) "
                         "if the HF hub is unreachable")
    ap.add_argument("--pairs_csv", default=None,
                    help="optional CSV with columns aff,neg for an extra Probe A")
    ap.add_argument("--out_dir", default="so_probe_results")
    ap.add_argument("--cache_dir", default="so_probe_cache")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch_size", type=int, default=64,
                    help="lower this (e.g. 16) if the GPU runs out of memory")
    ap.add_argument("--synthetic", action="store_true",
                    help="run the full pipeline on a toy encoder with a planted "
                         "frame subspace (no downloads)")
    args = ap.parse_args()

    if args.synthetic:
        enc = SyntheticEncoder()
        nevir = synthetic_nevir()
        geo, table = run_encoder(enc, nevir, [], [1, 2, 4], args.out_dir, None)
        print_report("synthetic (planted rank-2 frame)", geo, table)
        U, _, _ = fit_frame(*_synthetic_pairs(enc, nevir), 2)
        ang = np.degrees(np.arccos(np.clip(
            np.linalg.svd(U.T @ enc.U_true, compute_uv=False), -1, 1)))
        print(f"\nPlanted-subspace recovery, principal angles (deg): "
              f"{np.round(ang, 2).tolist()}  (near 0 = recovered)")
        return

    nevir = load_nevir(args.nevir_csv_dir)
    extra = load_pairs_csv(args.pairs_csv) if args.pairs_csv else []
    print(f"NevIR sizes: " + ", ".join(f"{k}={len(v)}" for k, v in nevir.items()))
    for name in args.models:
        print(f"\nLoading {name} ...")
        enc = STEncoder(name, device=args.device, batch_size=args.batch_size)
        geo, table = run_encoder(enc, nevir, extra, args.ranks, args.out_dir,
                                 args.cache_dir)
        print_report(name, geo, table)
        # release the model before loading the next one (avoids GPU OOM)
        del enc
        import gc
        gc.collect()
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    print(f"\nResults written to {os.path.abspath(args.out_dir)}")


def _synthetic_pairs(enc, nevir):
    texts = []
    for q1, q2, d1, d2 in nevir["train"]:
        texts += [d1, d2]
    H, _, _ = enc.encode(texts)
    return H[0::2], H[1::2]


if __name__ == "__main__":
    main()
