#!/usr/bin/env python3
"""Cross-encoder statistics for the subspace-removal (content readout) analysis.

For every model: NevIR full / polarity-removed / random-removed (exact McNemar,
pooled over seeds, with per-seed differences); SciFact content-vs-full on all,
negated and affirmative claims (query bootstrap 95% CI, Wilcoxon); training
damage negated-vs-affirmative difference-in-differences against the same
encoder's zero-shot run; ExcluIR content-vs-full.
"""
import json, sys, os
import numpy as np
from scipy.stats import binomtest, wilcoxon

R, E5PI, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
rng = np.random.default_rng(0)

def load(path):
    return {(d["mode"], d["seed"]): d["data"] for d in json.load(open(path))}

e5 = load(E5PI)
runs = {
    "E5 zero-shot":        ({("x", 1): e5[("zeroshot", 1)]}, None),
    "GTE zero-shot":       (load(f"{R}/enc_gte/per_instance.json"), None),
    "MPNet zero-shot":     (load(f"{R}/enc_mpnet/per_instance.json"), None),
    "Contriever zero-shot":(load(f"{R}/enc_contriever/per_instance.json"), None),
    "E5 plain FT":         ({k: v for k, v in e5.items() if k[0] == "plain_ft"}, "E5 zero-shot"),
    "E5 anchored frame":   ({k: v for k, v in e5.items() if k[0] == "frame"}, "E5 zero-shot"),
    "MPNet plain FT":      (load(f"{R}/ft_mpnet/per_instance.json"), "MPNet zero-shot"),
    "GTE plain FT":        (load(f"{R}/ft_gte_fp32/per_instance.json"), "GTE zero-shot"),
}

def seeds(name):
    return list(runs[name][0].values())

def mcn(name, ka, kb):
    W = L = 0; per = []
    for d in seeds(name):
        a = np.array(d[ka], bool); b = np.array(d[kb], bool)
        W += int((a & ~b).sum()); L += int((~a & b).sum()); per.append(a.mean() - b.mean())
    p = binomtest(W, W + L, .5).pvalue if W + L else 1.0
    return np.mean(per), per, W, L, p

def mean_of(name, k):
    return float(np.mean([np.mean(d[k]) for d in seeds(name)]))

def boot(d):
    b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
    return np.percentile(b, 2.5), np.percentile(b, 97.5)

def ndcg_diff(name, ka, kb, mask=None):
    D = np.array([np.array(d[ka], float) - np.array(d[kb], float) for d in seeds(name)]).mean(0)
    if mask is not None:
        D = D[mask]
    lo, hi = boot(D)
    nz = D[D != 0]
    p = wilcoxon(nz).pvalue if len(nz) > 10 else float("nan")
    return D.mean(), lo, hi, p

def did(name, zs):
    neg = np.array(seeds(name)[0]["scifact_negated"], bool)
    A = np.array([d["scifact_ndcg_q"] for d in seeds(name)], float).mean(0)
    Z = np.array(seeds(zs)[0]["scifact_ndcg_q"], float)
    d = A - Z
    iN, iA = np.where(neg)[0], np.where(~neg)[0]
    bs = np.array([d[rng.choice(iN, len(iN))].mean() - d[rng.choice(iA, len(iA))].mean()
                   for _ in range(5000)])
    obs = d[neg].mean() - d[~neg].mean()
    p = max(2 * min((bs <= 0).mean(), (bs >= 0).mean()), 1 / 5000)
    return d[neg].mean(), d[~neg].mean(), obs, np.percentile(bs, 2.5), np.percentile(bs, 97.5), p

out = {}
lines = []
P = lines.append
P("# Cross-encoder statistics: subspace removal (content readout)\n")
P("NevIR: exact McNemar pooled over seeds (per-seed differences in brackets). "
  "SciFact: per-query differences averaged over seeds, bootstrap 95% CI, Wilcoxon. "
  "The failed GTE run (ft_gte, NaN loss) is excluded; GTE plain FT = ft_gte_fp32.\n")
P("## A. NevIR pairwise accuracy\n")
P("| model | seeds | full | polarity removed | random removed | removed vs full: diff, W/L, p | removed vs random: diff, W/L, p |")
P("|---|---|---|---|---|---|---|")
for name in runs:
    f, c, r = (mean_of(name, k) for k in ("nevir_test", "nevir_test_content", "nevir_test_content_randU"))
    d1, per1, W1, L1, p1 = mcn(name, "nevir_test_content", "nevir_test")
    d2, per2, W2, L2, p2 = mcn(name, "nevir_test_content", "nevir_test_content_randU")
    out[name] = {"nevir_full": f, "nevir_content": c, "nevir_randU": r}
    P(f"| {name} | {len(seeds(name))} | {f:.3f} | {c:.3f} | {r:.3f} | {d1:+.3f} [{', '.join(f'{x:+.3f}' for x in per1)}], {W1}/{L1}, p={p1:.2g} | {d2:+.3f}, {W2}/{L2}, p={p2:.2g} |")

P("\n## B. SciFact: content readout vs full readout\n")
P("| model | all: diff [95% CI], p | negated (n=35): diff [CI], p | affirmative (n=265): diff [CI], p | content vs random-U, all: diff [CI], p |")
P("|---|---|---|---|---|")
for name in runs:
    neg = np.array(seeds(name)[0]["scifact_negated"], bool)
    cells = []
    for m in (None, neg, ~neg):
        d, lo, hi, p = ndcg_diff(name, "scifact_ndcg_q_content", "scifact_ndcg_q", m)
        cells.append(f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}], p={p:.2g}")
    d, lo, hi, p = ndcg_diff(name, "scifact_ndcg_q_content", "scifact_ndcg_q_content_randU")
    cells.append(f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}], p={p:.2g}")
    out[name].update({k: mean_of(name, k2) for k, k2 in (("sci_full", "scifact_ndcg_q"), ("sci_content", "scifact_ndcg_q_content"))})
    P(f"| {name} | " + " | ".join(cells) + " |")

P("\n## C. Training damage on SciFact relative to the same encoder zero-shot\n")
P("| model | drop on negated | drop on affirmative | difference (neg - aff) [95% CI] | p (bootstrap) |")
P("|---|---|---|---|---|")
for name, (_, zs) in runs.items():
    if zs is None:
        continue
    n, a, o, lo, hi, p = did(name, zs)
    out[name].update({"did": o, "did_lo": lo, "did_hi": hi})
    P(f"| {name} | {n:+.3f} | {a:+.3f} | {o:+.3f} [{lo:+.3f}, {hi:+.3f}] | {p:.2g} |")

P("\n## D. ExcluIR pairwise RR: content readout vs full\n")
P("| model | full | content | diff, W/L, p |")
P("|---|---|---|---|")
for name in runs:
    f, c = mean_of(name, "excluir_rr"), mean_of(name, "excluir_rr_content")
    d, per, W, L, p = mcn(name, "excluir_rr_content", "excluir_rr")
    P(f"| {name} | {f:.3f} | {c:.3f} | {d:+.4f}, {W}/{L}, p={p:.2g} |")

open(OUT, "w").write("\n".join(lines) + "\n")
json.dump(out, open(OUT.replace(".md", ".json"), "w"), indent=1)
print("\n".join(lines))
