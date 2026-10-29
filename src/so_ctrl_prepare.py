#!/usr/bin/env python3
"""
Dump the SciFact test claims needed to audit and rebuild the controlled-negation
set. CPU only, needs internet once (Hugging Face hub). Output: scifact_claims.json

For each of the 300 BEIR SciFact test claims:
  qid, text, naturally_negated (lexicon), rule output and rule type
  (insertion / prefix) of the original controlled-negation rule,
  relevant abstract ids with title and first 600 characters,
  and, where available, the original SciFact evidence labels
  (SUPPORT / CONTRADICT) from allenai/scifact.

Usage:  python so_ctrl_prepare.py
Requires semantic_odometry_probe.py and so_analysis.py in the same folder.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import load_dataset  # noqa: E402
from semantic_odometry_probe import n_cues  # noqa: E402
from so_analysis import negate_claim  # noqa: E402

corpus = load_dataset("BeIR/scifact", "corpus")["corpus"]
queries = load_dataset("BeIR/scifact", "queries")["queries"]
qrels = load_dataset("BeIR/scifact-qrels")["test"]
docs = {r["_id"]: (r["title"], r["text"]) for r in corpus}
rel = {}
for r in qrels:
    if r["score"] > 0:
        rel.setdefault(str(r["query-id"]), []).append(str(r["corpus-id"]))

labels = {}
for cfg in ("claims",):
    try:
        ds = load_dataset("allenai/scifact", cfg, trust_remote_code=True)
        for split in ds:
            for r in ds[split]:
                lab = r.get("evidence_label") or ""
                did = str(r.get("evidence_doc_id") or "")
                if lab and did:
                    labels.setdefault(str(r["id"]), {})[did] = lab
        print(f"evidence labels loaded for {len(labels)} claims")
    except Exception as e:  # labels are optional
        print(f"evidence labels not available ({e.__class__.__name__}); continuing without them")

out = []
for r in queries:
    q = str(r["_id"])
    if q not in rel:
        continue
    text = r["text"]
    neg = n_cues(text) > 0
    item = {"qid": q, "text": text, "naturally_negated": neg,
            "relevant": [{"doc_id": d, "title": docs[d][0], "abstract_start": docs[d][1][:600],
                          "label": labels.get(q, {}).get(d, "")} for d in rel[q] if d in docs]}
    if not neg:
        rule = negate_claim(text)
        item["rule_output"] = rule
        item["rule_type"] = "prefix" if rule.startswith("It is not the case that") else "insertion"
    out.append(item)

json.dump(out, open("scifact_claims.json", "w"), indent=1, ensure_ascii=False)
n_aff = sum(not x["naturally_negated"] for x in out)
n_pre = sum(x.get("rule_type") == "prefix" for x in out)
print(f"written scifact_claims.json: {len(out)} claims, {n_aff} affirmative "
      f"({n_aff - n_pre} insertion, {n_pre} prefix)")
