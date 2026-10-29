"""Split the controlled-negation penalty by rule type (insertion vs prefix) using scifact_claims.json
and the per-instance results of the revision runs."""
import json, sys, numpy as np
from scipy.stats import wilcoxon
C = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "controlled_negation/scifact_claims.json")); R = "results/revision_runs"
aff = [c for c in C if not c["naturally_negated"]]
typ = np.array([c["rule_type"] for c in aff]); rng = np.random.default_rng(3)
def boot(x):
    m = x[rng.integers(0, len(x), (5000, len(x)))].mean(1)
    return f"{x.mean():+.3f} [{np.percentile(m,2.5):+.3f}, {np.percentile(m,97.5):+.3f}]"
print(f"insertion {sum(typ=='insertion')}, prefix {sum(typ=='prefix')}")
for o in ["rv_zs_e5","rv_zs_mpnet","rv_zs_gte","rv_zs_contriever","rv_ft_e5","rv_ft_mpnet","rv_ft_gte"]:
    for r in json.load(open(f"{R}/{o}/per_instance.json")):
        d = r["data"]; neg = np.array(d["scifact_negated"])
        assert len(aff) == (~neg).sum()
        for t in ("insertion", "prefix"):
            m = typ == t
            pf = (np.array(d["an_full_ctrl_q"]) - np.array(d["an_full_sci_q"])[~neg])[m]
            pu = (np.array(d["an_U8_ctrl_q"]) - np.array(d["an_U8_sci_q"])[~neg])[m]
            print(f"{o} s{r['seed']} {t:9s} penalty {boot(pf)}  readout {boot(pu)}  change {boot(pu-pf)}")
