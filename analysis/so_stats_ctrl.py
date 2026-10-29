import json, numpy as np, sys
from scipy.stats import wilcoxon
R = sys.argv[1] if len(sys.argv) > 1 else "results/controlled_negation_run"
rng = np.random.default_rng(7); B = 5000
def boot(x, lo=2.5, hi=97.5):
    x = np.asarray(x, float); m = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(m, lo)), float(np.percentile(m, hi))
def wp(x):
    x = np.asarray(x, float); x = x[x != 0]; return float(wilcoxon(x).pvalue) if len(x) > 5 else 1.0
VARS = ["neg_v2", "v1_rule", "ctrl_matched", "ctrl_indeed", "ctrl_prefix"]
out = {}; zs = {}
for fam in ["e5", "mpnet"]:
    for r in json.load(open(f"{R}/ctrl_{fam}/per_instance.json")):
        d = r["data"]; tag = f"{fam} {r['mode']} s{r['seed']}"
        qpos = {q: i for i, q in enumerate(d["scifact_qids"])}
        rec = {}
        pen = {}
        for ro in ["full", "U8", "U8_plus_d", "rand8"]:
            orig = np.array(d[f"an_{ro}_sci_q"])
            for v in VARS:
                ids = d[f"var_{v}_qids"]; vs = np.array(d[f"an_{ro}_var_{v}_q"])
                pen[(ro, v)] = vs - orig[[qpos[q] for q in ids]]
        for v in VARS:
            rec[f"pen_full_{v}"] = boot(pen[("full", v)]) + (wp(pen[("full", v)]),)
            rec[f"pen_U8_{v}"] = boot(pen[("U8", v)]) + (wp(pen[("U8", v)]),)
            rec[f"change_U8_{v}"] = boot(pen[("U8", v)] - pen[("full", v)]) + (wp(pen[("U8", v)] - pen[("full", v)]),)
            rec[f"change_rand8_{v}"] = boot(pen[("rand8", v)] - pen[("full", v)]) + (wp(pen[("rand8", v)] - pen[("full", v)]),)
        for c in ["ctrl_matched", "ctrl_indeed", "ctrl_prefix"]:
            x = pen[("full", "neg_v2")] - pen[("full", c)]
            rec[f"neg_minus_{c}_full"] = boot(x) + (wp(x),)
            x = pen[("U8", "neg_v2")] - pen[("U8", c)]
            rec[f"neg_minus_{c}_U8"] = boot(x) + (wp(x),)
        if r["mode"] == "zeroshot":
            zs[fam] = pen
        else:
            for v in VARS:
                x = pen[("full", v)] - zs[fam][("full", v)]
                rec[f"ft_minus_zs_{v}"] = boot(x) + (wp(x),)
        for b in ["nfcorpus", "fiqa"]:
            f = np.array(d[f"an_full_{b}_q"])
            for ro in ["U8", "U8_plus_d", "rand8"]:
                x = np.array(d[f"an_{ro}_{b}_q"]) - f
                m, lo95, hi95 = boot(x); _, lo90, hi90 = boot(x, 5, 95)
                rec[f"pres_{b}_{ro}"] = (m, lo95, hi95, wp(x), lo90, hi90)
        out[tag] = rec
json.dump(out, open("results/statistics/stats_ctrl.json", "w"), indent=1)
for tag, rec in out.items():
    print(f"\n## {tag}")
    for k, v in rec.items():
        if k.startswith(("pen_", "change_U8", "neg_minus", "ft_minus", "change_rand8_neg")) or k.startswith("pres_"):
            print(f"  {k:32s} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}] p={v[3]:.2g}" + (f"  90%[{v[4]:+.3f},{v[5]:+.3f}]" if len(v) > 4 else ""))
