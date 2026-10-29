#!/usr/bin/env python3
"""Statistics for the Semantic Odometry three-seed run (tests3).

Binary outcomes (NevIR pairwise, ExcluIR RR): exact McNemar per seed and
pooled over seeds (discordant pairs summed), plus the mean difference.
SciFact (per-query nDCG@10): paired mean difference with a query-level
bootstrap 95% CI (queries resampled jointly across seeds) and a Wilcoxon
signed-rank test on per-query differences averaged over seeds.
"""
import json, sys
import numpy as np
from scipy.stats import binomtest, wilcoxon

PI, OUT = sys.argv[1], sys.argv[2]
pi = {(d["mode"], d["seed"]): d["data"] for d in json.load(open(PI))}
SEEDS = (1, 2, 3)
MODELS = ("plain_ft", "frame", "frame_randomU")
rng = np.random.default_rng(0)
rows = []

def key(m, s):
    return (m, 1) if m == "zeroshot" else (m, s)

def mcnemar_pooled(label, ma, ka, mb, kb):
    W = L = 0; diffs = []
    per = []
    for s in SEEDS:
        a = np.array(pi[key(ma, s)][ka], bool); b = np.array(pi[key(mb, s)][kb], bool)
        w, l = int((a & ~b).sum()), int((~a & b).sum()); W += w; L += l
        diffs.append(a.mean() - b.mean())
        per.append(f"{a.mean()-b.mean():+.3f}")
    p = binomtest(W, W + L, .5).pvalue
    rows.append((label, f"{np.mean(diffs):+.4f}", "[" + ", ".join(per) + "]",
                 f"{W}/{L}", f"{p:.2g}"))

def ndcg_pair(label, ma, ka, mb, kb, mask=None):
    A = np.array([pi[key(ma, s)][ka] for s in SEEDS], float)   # seeds x queries
    B = np.array([pi[key(mb, s)][kb] for s in SEEDS], float)
    D = A - B
    if mask is not None:
        D = D[:, mask]
    d = D.mean(0)                                                  # per-query, averaged over seeds
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
    nz = d[d != 0]
    p = wilcoxon(nz).pvalue if len(nz) > 10 else float("nan")
    per = "[" + ", ".join(f"{x:+.3f}" for x in D.mean(1)) + "]"
    rows.append((label, f"{d.mean():+.4f}", per,
                 f"95% CI [{np.percentile(boots,2.5):+.3f}, {np.percentile(boots,97.5):+.3f}]",
                 f"{p:.2g}"))

def did(label, m, k_after, k_before, mbase=None, kbase=None):
    """Difference-in-differences: (change on negated) - (change on affirmative)."""
    neg = np.array(pi[("zeroshot", 1)]["scifact_negated"], bool)
    A = np.array([pi[key(m, s)][k_after] for s in SEEDS], float)
    if mbase is None:
        B = np.array([pi[key(m, s)][k_before] for s in SEEDS], float)
    else:
        B = np.array([pi[key(mbase, s)][kbase] for s in SEEDS], float)
    d = (A - B).mean(0)
    stat = lambda x, n: x[n].mean() - x[~n].mean()
    obs = stat(d, neg)
    idx_n, idx_a = np.where(neg)[0], np.where(~neg)[0]
    boots = []
    for _ in range(5000):
        bn = rng.choice(idx_n, len(idx_n)); ba = rng.choice(idx_a, len(idx_a))
        boots.append(d[bn].mean() - d[ba].mean())
    boots = np.array(boots)
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    rows.append((label, f"{obs:+.4f}", f"neg {d[neg].mean():+.3f} / aff {d[~neg].mean():+.3f}",
                 f"95% CI [{np.percentile(boots,2.5):+.3f}, {np.percentile(boots,97.5):+.3f}]",
                 f"{max(p, 1/5000):.2g} (bootstrap)"))

def section(t):
    rows.append((f"## {t}", "", "", "", ""))

neg = np.array(pi[("zeroshot", 1)]["scifact_negated"], bool)

section("A. ExcluIR pairwise RR (McNemar, pooled over 3 seeds)")
for a, b in [("frame","zeroshot"),("frame","frame_randomU"),("frame","plain_ft"),
             ("plain_ft","zeroshot"),("frame_randomU","zeroshot")]:
    mcnemar_pooled(f"{a} vs {b}", a, "excluir_rr", b, "excluir_rr")

section("B. Reflection test, ExcluIR RR: reflect vs own (P1) and vs random-U reflect (P2)")
for m in ("zeroshot",) + MODELS:
    mcnemar_pooled(f"{m}: reflect vs own", m, "excluir_rr_reflect", m, "excluir_rr")
for m in MODELS:
    mcnemar_pooled(f"{m}: reflect vs reflect_randU", m, "excluir_rr_reflect", m, "excluir_rr_reflect_randU")
section("B3. Reflection, NevIR: reflect vs own (P3)")
for m in MODELS:
    mcnemar_pooled(f"{m}: reflect vs own (NevIR)", m, "nevir_test_reflect", m, "nevir_test")

section("C. SciFact damage from training (vs zeroshot), negated vs affirmative claims")
for m in MODELS:
    did(f"{m}: (drop on negated) - (drop on affirmative)", m, "scifact_ndcg_q", None,
        "zeroshot", "scifact_ndcg_q")
for m in MODELS:
    ndcg_pair(f"{m} vs zeroshot, all claims", m, "scifact_ndcg_q", "zeroshot", "scifact_ndcg_q")
    ndcg_pair(f"{m} vs zeroshot, negated claims (n={neg.sum()})", m, "scifact_ndcg_q", "zeroshot", "scifact_ndcg_q", neg)
    ndcg_pair(f"{m} vs zeroshot, affirmative claims (n={(~neg).sum()})", m, "scifact_ndcg_q", "zeroshot", "scifact_ndcg_q", ~neg)

section("D. Content readout on SciFact: content vs full (C1), content vs random-U (C2)")
for m in ("zeroshot",) + MODELS:
    ndcg_pair(f"{m}: content vs full, all", m, "scifact_ndcg_q_content", m, "scifact_ndcg_q")
    ndcg_pair(f"{m}: content vs full, negated", m, "scifact_ndcg_q_content", m, "scifact_ndcg_q", neg)
    ndcg_pair(f"{m}: content vs full, affirmative", m, "scifact_ndcg_q_content", m, "scifact_ndcg_q", ~neg)
    ndcg_pair(f"{m}: content vs content_randU, all", m, "scifact_ndcg_q_content", m, "scifact_ndcg_q_content_randU")
for m in ("frame", "plain_ft"):
    did(f"{m}: recovery on negated - recovery on affirmative", m, "scifact_ndcg_q_content", "scifact_ndcg_q")
ndcg_pair("frame(content) vs plain_ft(content), all (C4)", "frame", "scifact_ndcg_q_content", "plain_ft", "scifact_ndcg_q_content")

section("E. Content readout, NevIR: content vs full (C3) and vs random-U")
for m in ("zeroshot",) + MODELS:
    mcnemar_pooled(f"{m}: content vs full (NevIR)", m, "nevir_test_content", m, "nevir_test")
    mcnemar_pooled(f"{m}: content vs content_randU (NevIR)", m, "nevir_test_content", m, "nevir_test_content_randU")

section("F. Content readout, ExcluIR: does the exclusion gain survive removing U?")
for m in MODELS:
    mcnemar_pooled(f"{m}: content vs full (ExcluIR)", m, "excluir_rr_content", m, "excluir_rr")
mcnemar_pooled("frame(content) vs plain_ft(full) (ExcluIR)", "frame", "excluir_rr_content", "plain_ft", "excluir_rr")

with open(OUT, "w") as f:
    f.write("| comparison | mean diff | per seed / by group | CI or win/loss | p |\n|---|---|---|---|---|\n")
    for r in rows:
        f.write("| " + " | ".join(r) + " |\n")
for r in rows:
    print(" | ".join(r))
