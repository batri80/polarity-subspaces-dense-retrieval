#!/usr/bin/env python3
"""
Revision analysis package for "Polarity Subspaces in Dense Retrieval".

Runs after a zero-shot or plain fine-tuned model is ready (called from
so_train.py with --analysis_pack). Every text is embedded ONCE (raw mean-pooled
embedding h); every readout below is then a linear projection of h, so all
ranks, baselines and controls cost only matrix arithmetic.

Readouts (each = normalize((I - V V^T) h) for an orthonormal V, or the full h):
  full            no projection
  U_k             offset-aware polarity subspace, k in RANKS (Eq. 1 fit)
  rand_k          random orthonormal subspace of rank k (control)
  U8_plus_d       U_8 augmented with the cue-offset direction d (rank 9)
  d_only          the cue-offset direction alone (rank 1)
  meandiff        normalized mean(neg - aff) direction (simplest baseline)
  probe           ridge linear-probe direction separating aff from neg
  pca8            top-8 principal components of the pair embeddings
Diagnostics:
  stability       U_8 refit on 5 random halves; subspace overlap with U_8
  heldout_fit     Eq. 1 on held-out NevIR test pairs: variance of (b - a)
                  explained, vs a translation-only model, for each k
  readout_residual  ||(I-VV^T)(b-a)|| / ||b-a|| on held-out pairs: how much
                  of an aff/neg difference survives each readout (offset issue)
Evaluations per readout: NevIR test (all / explicit-cue / implicit pairs),
SciFact (all / negated / affirmative claims), controlled negation of the
affirmative SciFact claims (same corpus and labels, only polarity changed),
ExcluIR pairwise RR, and optional BEIR preservation sets (NFCorpus, FiQA).
"""
import math
import re

import numpy as np
import torch

from semantic_odometry_probe import fit_frame, n_cues, orient_pairs

RANKS = (1, 2, 4, 8, 16, 32)
AUX = {"is", "are", "was", "were", "can", "could", "does", "do", "did", "has",
       "have", "had", "will", "would", "may", "might", "should", "must", "shall"}


# ---------------------------------------------------------------------------
# Controlled negation (rule fixed 28 Oct 2026, before any result)
# ---------------------------------------------------------------------------
def negate_claim(text):
    """Insert 'not' after the first auxiliary or copula; otherwise prefix
    'It is not the case that'. Applied only to claims without a negation cue,
    so the output always carries exactly one added cue."""
    m = re.search(r"\b(" + "|".join(sorted(AUX)) + r")\b", text, flags=re.IGNORECASE)
    if m:
        return text[:m.end()] + " not" + text[m.end():]
    return "It is not the case that " + text[0].lower() + text[1:]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def raw_embed(model, texts, role, bs):
    model.eval()
    out = []
    for i in range(0, len(texts), bs):
        out.append(model(texts[i:i + bs], role, return_raw=True)[1].float().cpu())
    return torch.cat(out).numpy().astype(np.float32)


def orth(M):
    q, _ = np.linalg.qr(M)
    return q.astype(np.float32)


def readout(H, V):
    X = H if V is None else H - (H @ V) @ V.T
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def nevir_correct(Q1, Q2, D1, D2):
    s = lambda a, b: (a * b).sum(1)
    return (s(Q1, D1) > s(Q1, D2)) & (s(Q2, D2) > s(Q2, D1))


def ndcg10_per_query(Q, D, qids, dids, rel):
    S = Q @ D.T
    k = min(10, S.shape[1])
    top = np.argpartition(-S, k - 1, axis=1)[:, :k]
    order = np.take_along_axis(top, np.argsort(-np.take_along_axis(S, top, 1), 1), 1)
    out = []
    for i, q in enumerate(qids):
        r = rel[q]
        dcg = sum(1 / math.log2(j + 2) for j, di in enumerate(order[i]) if dids[di] in r)
        idcg = sum(1 / math.log2(j + 2) for j in range(min(len(r), 10)))
        out.append(dcg / idcg if idcg else 0.0)
    return np.array(out)


def subspace_overlap(U1, U2):
    """Mean squared cosine of principal angles: 1 = identical, k/D = chance."""
    return float(np.linalg.norm(U1.T @ U2) ** 2 / min(U1.shape[1], U2.shape[1]))


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------
def analysis_pack(args, model, mode, data, res, per_instance, log=print):
    if model.uses_frame:
        raise ValueError("analysis_pack is defined for zero-shot and plain fine-tuned models")
    bs = args.eval_batch_size
    rng = np.random.default_rng(2026)
    nev = data["nevir"]
    A_ = {}

    # --- fitting pairs: NevIR TRAIN (same procedure as probe_init_U) --------
    pq, pd = [], []
    for q1, q2, d1, d2 in nev["train"]:
        pq += orient_pairs([(q1, q2)])
        pd += orient_pairs([(d1, d2)])
    A = np.vstack([raw_embed(model, [a for a, _ in pd], "doc", bs),
                   raw_embed(model, [a for a, _ in pq], "query", bs)])
    B = np.vstack([raw_embed(model, [b for _, b in pd], "doc", bs),
                   raw_embed(model, [b for _, b in pq], "query", bs)])
    D = A.shape[1]
    log(f"    analysis: {len(A)} fitting pairs, dimension {D}")

    fits = {}
    for k in RANKS:
        U, c, d = fit_frame(A, B, k)
        fits[k] = (U.astype(np.float32), float(c), d.astype(np.float32))
    U8, c8, d8 = fits[8]
    dhat = (d8 / (np.linalg.norm(d8) + 1e-12))[:, None]
    diff = B - A
    md = diff.mean(0)
    X = np.vstack([A, B]); X = X - X.mean(0)
    y = np.r_[-np.ones(len(A)), np.ones(len(B))]
    w = np.linalg.solve(X.T @ X + 1.0 * np.eye(D), X.T @ y)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)

    V = {"full": None}
    for k in RANKS:
        V[f"U{k}"] = fits[k][0]
        V[f"rand{k}"] = orth(rng.standard_normal((D, k)))
    V["U8_plus_d"] = orth(np.hstack([U8, dhat]))
    V["d_only"] = dhat.astype(np.float32)
    V["meandiff"] = (md / np.linalg.norm(md))[:, None].astype(np.float32)
    V["probe"] = (w / np.linalg.norm(w))[:, None].astype(np.float32)
    V["pca8"] = Vt[:8].T.astype(np.float32)

    # --- diagnostics: stability, held-out fit, readout residual -------------
    stab = []
    for b in range(5):
        idx = rng.choice(len(A), len(A) // 2, replace=False)
        Ub, _, _ = fit_frame(A[idx], B[idx], 8)
        stab.append(subspace_overlap(Ub.astype(np.float32), U8))
        V[f"U8_half{b}"] = Ub.astype(np.float32)
    A_["stability_overlap"] = {"mean": float(np.mean(stab)), "values": stab,
                               "chance": 8.0 / D}
    log(f"    stability: overlap of half-sample U8 with full U8 = "
        f"{np.mean(stab):.3f} (chance {8.0 / D:.3f})")

    tq, td = [], []
    for q1, q2, d1, d2 in nev["test"]:
        tq += orient_pairs([(q1, q2)])
        td += orient_pairs([(d1, d2)])
    At = np.vstack([raw_embed(model, [a for a, _ in td], "doc", bs),
                    raw_embed(model, [a for a, _ in tq], "query", bs)])
    Bt = np.vstack([raw_embed(model, [b for _, b in td], "doc", bs),
                    raw_embed(model, [b for _, b in tq], "query", bs)])
    dt = Bt - At
    energy = float((dt ** 2).sum())
    hf = {"translation_only": 1 - float(((dt - d8) ** 2).sum()) / energy}
    for k in RANKS:
        U, c, d = fits[k]
        pred = d - c * (At @ U) @ U.T
        hf[f"k{k}"] = 1 - float(((dt - pred) ** 2).sum()) / energy
    A_["heldout_fit_explained"] = hf
    rr = {}
    for name in ("U8", "U8_plus_d", "d_only", "rand8", "meandiff", "pca8"):
        P = dt - (dt @ V[name]) @ V[name].T
        rr[name] = float(np.median(np.linalg.norm(P, axis=1) / np.linalg.norm(dt, axis=1)))
    A_["readout_residual_median"] = rr
    log("    held-out fit (share of aff/neg difference explained): " +
        ", ".join(f"{k}={v:.3f}" for k, v in hf.items()))
    log("    residual after readout (median share of aff/neg difference kept): " +
        ", ".join(f"{k}={v:.3f}" for k, v in rr.items()))

    # --- evaluation embeddings (once each) ---------------------------------
    rows = nev["test"]
    E = {"q1": raw_embed(model, [r[0] for r in rows], "query", bs),
         "q2": raw_embed(model, [r[1] for r in rows], "query", bs),
         "d1": raw_embed(model, [r[2] for r in rows], "doc", bs),
         "d2": raw_embed(model, [r[3] for r in rows], "doc", bs)}
    explicit = np.array([n_cues(r[2]) != n_cues(r[3]) for r in rows])
    per_instance["nevir_explicit"] = explicit.tolist()
    A_["nevir_explicit_share"] = float(explicit.mean())

    sci = None
    if data.get("scifact"):
        docs, qs, rel = data["scifact"]
        dids, qids = list(docs), list(qs)
        neg = np.array([n_cues(qs[q]) > 0 for q in qids])
        aff_ids = [q for q, n in zip(qids, neg) if not n]
        ctrl_txt = {q: negate_claim(qs[q]) for q in aff_ids}
        sci = {"D": raw_embed(model, [docs[d] for d in dids], "doc", bs),
               "Q": raw_embed(model, [qs[q] for q in qids], "query", bs),
               "C": raw_embed(model, [ctrl_txt[q] for q in aff_ids], "query", bs),
               "dids": dids, "qids": qids, "neg": neg, "aff_ids": aff_ids, "rel": rel}
        per_instance["scifact_negated"] = neg.tolist()
        A_["scifact_negated_claims"] = [qs[q] for q, n in zip(qids, neg) if n]
        A_["controlled_negation_examples"] = [[qs[q], ctrl_txt[q]] for q in aff_ids[:25]]
        A_["controlled_negation_rule_share"] = float(np.mean(
            [not t.startswith("It is not the case that") for t in ctrl_txt.values()]))
        # optional verified claim variants: JSON list of {"qid":..., "variants": {name: text}}
        sci["var"] = {}
        if getattr(args, "ctrl_file", None):
            import json as _json
            items = _json.load(open(args.ctrl_file))
            names = sorted({v for it in items for v in it["variants"]})
            for v in names:
                ids = [str(it["qid"]) for it in items if v in it["variants"] and str(it["qid"]) in qs]
                txt = {str(it["qid"]): it["variants"][v] for it in items if v in it["variants"]}
                sci["var"][v] = (ids, raw_embed(model, [txt[q] for q in ids], "query", bs))
                per_instance[f"var_{v}_qids"] = ids
            log(f"    claim variants from {args.ctrl_file}: " +
                ", ".join(f"{v} ({len(sci['var'][v][0])})" for v in names))

    exc = None
    if data.get("excluir"):
        xr = data["excluir"][0]
        exc = {"q": raw_embed(model, [r[0] for r in xr], "query", bs),
               "p": raw_embed(model, [r[1] for r in xr], "doc", bs),
               "n": raw_embed(model, [r[2] for r in xr], "doc", bs)}

    beir = {}
    for name, (docs, qs, rel) in data.get("analysis_beir", {}).items():
        dids, qids = list(docs), list(qs)
        log(f"    embedding {name}: {len(dids)} documents, {len(qids)} queries")
        beir[name] = (raw_embed(model, [docs[d] for d in dids], "doc", bs),
                      raw_embed(model, [qs[q] for q in qids], "query", bs), dids, qids, rel)

    # --- evaluate every readout --------------------------------------------
    keep_pi = {"full", "U8", "U8_plus_d", "d_only", "rand8", "meandiff", "probe", "pca8"}
    if getattr(args, "analysis_light", False):
        V = {k: V[k] for k in ("full", "U8", "U8_plus_d", "d_only", "rand8")}
    table = {}
    for name, VV in V.items():
        r = {}
        nc = nevir_correct(*(readout(E[x], VV) for x in ("q1", "q2", "d1", "d2")))
        r["nevir"] = float(nc.mean())
        r["nevir_explicit"] = float(nc[explicit].mean()) if explicit.any() else float("nan")
        r["nevir_implicit"] = float(nc[~explicit].mean()) if (~explicit).any() else float("nan")
        if name in keep_pi:
            per_instance[f"an_{name}_nevir"] = nc.tolist()
        quick = name.startswith("U8_half")
        if sci is not None and not quick:
            Dn = readout(sci["D"], VV)
            s = ndcg10_per_query(readout(sci["Q"], VV), Dn, sci["qids"], sci["dids"], sci["rel"])
            c = ndcg10_per_query(readout(sci["C"], VV), Dn, sci["aff_ids"], sci["dids"], sci["rel"])
            r["sci"] = float(s.mean())
            r["sci_neg"] = float(s[sci["neg"]].mean()) if sci["neg"].any() else float("nan")
            r["sci_aff"] = float(s[~sci["neg"]].mean())
            r["sci_ctrl_negated"] = float(c.mean())
            r["sci_ctrl_penalty"] = float(c.mean() - s[~sci["neg"]].mean())
            if name in keep_pi:
                per_instance[f"an_{name}_sci_q"] = s.tolist()
                per_instance[f"an_{name}_ctrl_q"] = c.tolist()
            for v, (ids, Qv) in sci.get("var", {}).items():
                sv = ndcg10_per_query(readout(Qv, VV), Dn, ids, sci["dids"], sci["rel"])
                r[f"var_{v}"] = float(sv.mean())
                if name in keep_pi:
                    per_instance[f"an_{name}_var_{v}_q"] = sv.tolist()
        if exc is not None and not quick:
            q, p, n = (readout(exc[x], VV) for x in ("q", "p", "n"))
            ex = (q * p).sum(1) > (q * n).sum(1)
            r["excluir_rr"] = float(ex.mean())
            if name in keep_pi:
                per_instance[f"an_{name}_excl"] = ex.tolist()
        for bname, (Db, Qb, dids, qids, rel) in beir.items():
            if quick or name.startswith("rand") and name != "rand8":
                continue
            sb = ndcg10_per_query(readout(Qb, VV), readout(Db, VV), qids, dids, rel)
            r[bname] = float(sb.mean())
            if name in keep_pi:
                per_instance[f"an_{name}_{bname}_q"] = sb.tolist()
        table[name] = r
    A_["readouts"] = table
    res["analysis"] = A_

    # compact console report
    cols = ["nevir", "nevir_explicit", "nevir_implicit", "sci", "sci_neg",
            "sci_ctrl_negated", "sci_ctrl_penalty", "excluir_rr"] + list(beir)
    log("    readout          " + "".join(f"{c[:14]:>15}" for c in cols))
    for name, r in table.items():
        if name.startswith("U8_half"):
            continue
        log(f"    {name:<16} " + "".join(
            f"{r.get(c, float('nan')):>15.3f}" for c in cols))
