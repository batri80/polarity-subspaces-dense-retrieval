"""Statistics for the revision runs (rv_*): per-seed exact McNemar, query bootstrap, Wilcoxon."""
import json, sys, numpy as np
from scipy.stats import binomtest, wilcoxon
R = sys.argv[1] if len(sys.argv) > 1 else "results/revision_runs"
rng = np.random.default_rng(0); B = 5000
def load(o): return json.load(open(f"{R}/{o}/per_instance.json")), json.load(open(f"{R}/{o}/results.json"))
def mcn(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    p = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return float(b.mean() - a.mean()), n10, n01, float(p)
def boot(x):
    x = np.asarray(x, float); idx = rng.integers(0, len(x), (B, len(x)))
    m = x[idx].mean(1); return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
def wil(x):
    x = np.asarray(x, float); return float(wilcoxon(x[x != 0]).pvalue) if (x != 0).sum() > 5 else 1.0
out = {}; lines = []
P = lambda s: (lines.append(s), print(s))
zs_ctrl = {}
for o in ["rv_zs_e5", "rv_zs_mpnet", "rv_zs_gte", "rv_zs_contriever", "rv_ft_e5", "rv_ft_mpnet", "rv_ft_gte"]:
    pis, ress = load(o)
    for pi, res in zip(pis, ress):
        d = pi["data"]; tag = f"{o} s{pi['seed']}"; neg = np.array(d["scifact_negated"]); ex = np.array(d["nevir_explicit"])
        rec = {}
        for k in ["U8", "rand8", "U8_plus_d", "d_only", "meandiff", "probe", "pca8"]:
            rec[f"nevir_{k}"] = mcn(d["an_full_nevir"], d[f"an_{k}_nevir"])
            rec[f"nevir_{k}_vs_U8"] = mcn(d[f"an_{k}_nevir"], d["an_U8_nevir"])
        for nm, msk in [("explicit", ex), ("implicit", ~ex)]:
            rec[f"nevir_U8_{nm}"] = mcn(np.array(d["an_full_nevir"])[msk], np.array(d["an_U8_nevir"])[msk])
        rec["excl_U8"] = mcn(d["an_full_excl"], d["an_U8_excl"])
        sf, s8, sr = (np.array(d[f"an_{k}_sci_q"]) for k in ["full", "U8", "rand8"])
        rec["sci_U8"] = boot(s8 - sf) + (wil(s8 - sf),)
        rec["sci_U8_neg"] = boot((s8 - sf)[neg]) + (wil((s8 - sf)[neg]),)
        rec["sci_U8_aff"] = boot((s8 - sf)[~neg]) + (wil((s8 - sf)[~neg]),)
        rec["sci_U8_vs_rand8"] = boot(s8 - sr) + (wil(s8 - sr),)
        pen = {}
        for k in ["full", "U8", "U8_plus_d", "rand8"]:
            pen[k] = np.array(d[f"an_{k}_ctrl_q"]) - np.array(d[f"an_{k}_sci_q"])[~neg]
            rec[f"ctrl_penalty_{k}"] = boot(pen[k]) + (wil(pen[k]),)
        rec["ctrl_penalty_change_U8"] = boot(pen["U8"] - pen["full"]) + (wil(pen["U8"] - pen["full"]),)
        rec["ctrl_penalty_change_rand8"] = boot(pen["rand8"] - pen["full"]) + (wil(pen["rand8"] - pen["full"]),)
        fam = o.split("_")[-1]
        if o.startswith("rv_zs"): zs_ctrl[fam] = pen["full"]
        else:
            rec["ctrl_penalty_ft_minus_zs"] = boot(pen["full"] - zs_ctrl[fam]) + (wil(pen["full"] - zs_ctrl[fam]),)
        a = res["analysis"]
        rec["stability"] = a["stability_overlap"]; rec["fit"] = a["heldout_fit_explained"]; rec["resid"] = a["readout_residual_median"]
        rec["readouts"] = a["readouts"]
        out[tag] = rec
        P(f"\n## {tag}")
        for k, v in rec.items():
            if k in ("readouts", "stability", "fit", "resid"): continue
            if len(v) == 4 and isinstance(v[1], int): P(f"  {k:28s} diff {v[0]:+.3f}  (lost {v[1]}, gained {v[2]})  p={v[3]:.2g}")
            else: P(f"  {k:28s} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]  Wilcoxon p={v[3]:.2g}")
json.dump(out, open("results/statistics/stats_rv.json", "w"), indent=1, default=float)
open("results/statistics/stats_rv.md", "w").write("\n".join(lines))
