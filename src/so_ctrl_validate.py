#!/usr/bin/env python3
"""
Automated validation of the controlled-negation claim set with independent,
non-generative models (no model that produced the sentences is used to judge them).

  Fidelity + scope : two NLI cross-encoders judge (original -> negated) and
                     (original -> matched control). A valid negation should be
                     classified CONTRADICTION; a valid control ENTAILMENT.
  Grammaticality   : a CoLA acceptability classifier scores the original, the
                     negated claim and the control; a drop relative to the
                     original flags a likely grammatical problem.

Input : ctrl_v2.json   (from the audit step)
Output: ctrl_validation.json  (per claim scores and flags) and a printed summary.
CPU is enough (about 10 minutes); needs internet once to download the models.
"""
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

NLI = ["microsoft/deberta-large-mnli", "roberta-large-mnli"]
COLA = "textattack/roberta-base-CoLA"
D = json.load(open("ctrl_v2.json"))
items = [(i, o) for i, o in enumerate(D) if o.get("neg_v2")]
dev = "cuda" if torch.cuda.is_available() else "cpu"


def nli_scores(name, pairs, bs=16):
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForSequenceClassification.from_pretrained(name).to(dev).eval()
    lab = {v.lower(): k for k, v in m.config.id2label.items()}
    out = []
    for s in range(0, len(pairs), bs):
        a, b = zip(*pairs[s:s + bs])
        x = tok(list(a), list(b), padding=True, truncation=True, return_tensors="pt").to(dev)
        with torch.no_grad():
            p = m(**x).logits.softmax(-1).cpu()
        out += [{"contradiction": float(r[lab["contradiction"]]),
                 "entailment": float(r[lab["entailment"]]),
                 "neutral": float(r[lab["neutral"]])} for r in p]
    return out


def cola_scores(texts, bs=32):
    tok = AutoTokenizer.from_pretrained(COLA)
    m = AutoModelForSequenceClassification.from_pretrained(COLA).to(dev).eval()
    out = []
    for s in range(0, len(texts), bs):
        x = tok(texts[s:s + bs], padding=True, truncation=True, return_tensors="pt").to(dev)
        with torch.no_grad():
            out += m(**x).logits.softmax(-1)[:, 1].cpu().tolist()   # P(acceptable)
    return out


orig = [o["original"] for _, o in items]
neg = [o["neg_v2"] for _, o in items]
ctl = [o["ctrl_v2"] for _, o in items]
res = {i: {"qid": o["qid"]} for i, o in items}
for name in NLI:
    short = name.split("/")[-1]
    print("NLI", name)
    for key, hyp in (("neg", neg), ("ctrl", ctl)):
        for (i, _), sc in zip(items, nli_scores(name, list(zip(orig, hyp)))):
            res[i][f"{short}_{key}"] = sc
print("CoLA", COLA)
for key, texts in (("orig", orig), ("neg", neg), ("ctrl", ctl)):
    for (i, _), p in zip(items, cola_scores(texts)):
        res[i][f"cola_{key}"] = p

short = [n.split("/")[-1] for n in NLI]
summary = {}
for i, r in res.items():
    negok = [max(r[f"{s}_neg"], key=r[f"{s}_neg"].get) == "contradiction" for s in short]
    ctlok = [max(r[f"{s}_ctrl"], key=r[f"{s}_ctrl"].get) == "entailment" for s in short]
    r["neg_contradiction_votes"] = sum(negok)
    r["ctrl_entailment_votes"] = sum(ctlok)
    r["grammar_drop_neg"] = r["cola_orig"] - r["cola_neg"]
    r["grammar_drop_ctrl"] = r["cola_orig"] - r["cola_ctrl"]
    r["flag"] = (sum(negok) < 2) or (sum(ctlok) < 2) or r["grammar_drop_neg"] > 0.3 or r["grammar_drop_ctrl"] > 0.3
n = len(res)
summary["n"] = n
summary["negation judged contradiction by both NLI models"] = sum(r["neg_contradiction_votes"] == 2 for r in res.values())
summary["negation judged contradiction by at least one"] = sum(r["neg_contradiction_votes"] >= 1 for r in res.values())
summary["control judged entailment by both"] = sum(r["ctrl_entailment_votes"] == 2 for r in res.values())
summary["grammar drop > 0.3 (negation)"] = sum(r["grammar_drop_neg"] > 0.3 for r in res.values())
summary["grammar drop > 0.3 (control)"] = sum(r["grammar_drop_ctrl"] > 0.3 for r in res.values())
summary["flagged for review"] = sum(r["flag"] for r in res.values())
for k, v in summary.items():
    print(f"  {k}: {v}")
json.dump({"summary": summary, "items": res}, open("ctrl_validation.json", "w"), indent=1)
print("written ctrl_validation.json")
