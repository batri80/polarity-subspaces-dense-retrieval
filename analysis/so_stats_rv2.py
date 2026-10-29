import json, numpy as np
R="results/revision_runs"; rng=np.random.default_rng(1); B=5000
def pi(o): return json.load(open(f"{R}/{o}/per_instance.json"))
out={}
for fam in ["e5","mpnet","gte"]:
    z=pi(f"rv_zs_{fam}")[0]["data"]; neg=np.array(z["scifact_negated"])
    zs=np.array(z["an_full_sci_q"]); zpen=np.array(z["an_full_ctrl_q"])-zs[~neg]
    rows=[]
    for r in pi(f"rv_ft_{fam}"):
        d=r["data"]; f=np.array(d["an_full_sci_q"]); u=np.array(d["an_U8_sci_q"])
        dl=f-zs
        def did(x):
            a,b=x[neg],x[~neg]
            m=[rng.choice(a,len(a)).mean()-rng.choice(b,len(b)).mean() for _ in range(B)]
            return float(a.mean()-b.mean()),float(np.percentile(m,2.5)),float(np.percentile(m,97.5))
        pen_f=np.array(d["an_full_ctrl_q"])-f[~neg]; pen_u=np.array(d["an_U8_ctrl_q"])-u[~neg]
        rec=dict(seed=r["seed"], zs_neg=zs[neg].mean(), zs_aff=zs[~neg].mean(), ft_neg=f[neg].mean(), ft_aff=f[~neg].mean(),
                 ro_neg=u[neg].mean(), ro_aff=u[~neg].mean(), damage_did=did(dl), recovery_did=did(u-f),
                 pen_zs=zpen.mean(), pen_ft=pen_f.mean(), pen_ro=pen_u.mean(),
                 ctrl_zs=np.array(z["an_full_ctrl_q"]).mean(), ctrl_ft=np.array(d["an_full_ctrl_q"]).mean(), ctrl_ro=np.array(d["an_U8_ctrl_q"]).mean(),
                 ro_vs_zs_all=float(u.mean()-zs.mean()))
        rows.append(rec)
        print(fam, {k:(round(v,3) if isinstance(v,float) else (tuple(round(x,3) for x in v) if isinstance(v,tuple) else v)) for k,v in rec.items()})
    out[fam]=rows
json.dump(out,open("results/statistics/stats_rv_groups.json","w"),indent=1,default=float)
