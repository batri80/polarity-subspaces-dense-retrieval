import json, spacy
nlp = spacy.load("en_core_web_sm")
C = json.load(open("scifact_claims.json"))
DO = {"VBZ": "does", "VBP": "do", "VBD": "did"}
out = []
for c in C:
    if c["naturally_negated"]:
        continue
    t = c["text"].strip(); doc = nlp(t)
    root = [w for w in doc if w.dep_ == "ROOT"][0]
    auxs = [w for w in root.children if w.dep_ in ("aux", "auxpass") and w.i < root.i]
    flag = []
    if any(w.dep_ == "conj" and w.pos_ in ("VERB", "AUX") for w in root.children):
        flag.append("coordinated predicate")
    neg = ctl = None; method = ""
    if auxs:
        a = auxs[0]
        neg = t[:a.idx + len(a.text)] + " not" + t[a.idx + len(a.text):]
        ctl = t[:a.idx + len(a.text)] + " indeed" + t[a.idx + len(a.text):]
        method = "aux"
    elif root.lemma_ == "be" and root.tag_ in ("VBZ", "VBP", "VBD"):
        neg = t[:root.idx + len(root.text)] + " not" + t[root.idx + len(root.text):]
        ctl = t[:root.idx + len(root.text)] + " indeed" + t[root.idx + len(root.text):]
        method = "copula"
    elif root.tag_ in DO and root.pos_ in ("VERB", "AUX"):
        lem = root.lemma_
        neg = t[:root.idx] + DO[root.tag_] + " not " + lem + t[root.idx + len(root.text):]
        ctl = t[:root.idx] + DO[root.tag_] + " " + lem + t[root.idx + len(root.text):]
        method = "do-support"
    else:
        flag.append(f"no finite main verb (root '{root.text}' {root.tag_})")
        method = "manual"
    out.append({"qid": c["qid"], "original": t, "rule_v1": c["rule_output"], "rule_v1_type": c["rule_type"],
                "neg_v2": neg, "ctrl_v2": ctl, "method": method, "flags": flag})
json.dump(out, open("draft_v2.json", "w"), indent=1, ensure_ascii=False)
import collections; print(collections.Counter(o["method"] for o in out), sum(bool(o["flags"]) for o in out))
