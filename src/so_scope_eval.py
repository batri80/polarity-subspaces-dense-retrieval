#!/usr/bin/env python3
"""
Evaluate the lexical negation cue/scope detector against BioScope gold
annotations (Vincze et al., 2008), replacing manual annotation.

Reports, per BioScope sub-corpus (abstracts, full papers, clinical):
  * sentence-level cue detection: does the sentence contain a negation?
    precision / recall / F1 for the training detector and the extended one
  * token-level scope: tokens inside a gold negation scope (cue tokens
    excluded, *SEM convention) vs tokens our detector marks in scope;
    precision / recall / F1 over sentences that contain a gold negation
  * errors by type: false-positive and false-negative cue words, most common

Usage (needs internet once to fetch the corpus, or pass a local copy):
  python so_scope_eval.py                         # downloads bioscope.zip
  python so_scope_eval.py --zip path/to/bioscope.zip
Requires semantic_odometry_probe.py and so_train.py in the same folder.
"""
import argparse
import collections
import io
import json
import os
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import semantic_odometry_probe as S  # noqa: E402
import so_train as T  # noqa: E402

URLS = [  # the corpus moved from rgai.sed.hu (certificate no longer matches) to rgai.inf.u-szeged.hu
    "https://rgai.inf.u-szeged.hu/sites/rgai.inf.u-szeged.hu/files/bioscope.zip",
    "https://rgai.sed.hu/sites/rgai.sed.hu/files/bioscope.zip",
]


def flatten(sent):
    """Return sentence text and gold (start, end) spans of negation cues and of
    the scopes those cues govern."""
    text, cues, scopes = [], [], {}

    def walk(el):
        start = sum(len(t) for t in text)
        if el.text:
            text.append(el.text)
        for ch in el:
            walk(ch)
            if ch.tail:
                text.append(ch.tail)
        end = sum(len(t) for t in text)      # before the parent appends el.tail
        if el.tag == "cue" and el.get("type") == "negation":
            cues.append((start, end, el.get("ref")))
        if el.tag == "xcope":
            scopes[el.get("id")] = (start, end)

    # walk children manually so tails are attributed correctly
    if sent.text:
        text.append(sent.text)
    for ch in sent:
        walk(ch)
        if ch.tail:
            text.append(ch.tail)
    full = "".join(text)
    neg_scopes = [scopes[r] for _, _, r in cues if r in scopes]
    return full, [(a, b) for a, b, _ in cues], neg_scopes


def tokens(text):
    return [(m.start(), m.end(), m.group(0)) for m in S.WORD_RE.finditer(text)]


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if p == p and r == r and p + r else float("nan")
    return p, r, f


def evaluate(sentences, detector):
    """detector(text) -> list of (start, end, parity, is_cue) like word_parities."""
    cue_tp = cue_fp = cue_fn = 0
    sc_tp = sc_fp = sc_fn = 0
    fp_words, fn_words = collections.Counter(), collections.Counter()
    for text, gcues, gscopes in sentences:
        words = detector(text)
        pred_has = any(c for *_, c in words)
        gold_has = bool(gcues)
        cue_tp += pred_has and gold_has
        cue_fp += pred_has and not gold_has
        cue_fn += gold_has and not pred_has
        if pred_has and not gold_has:
            fp_words.update(text[a:b].lower() for a, b, _, c in words if c)
        if gold_has and not pred_has:
            fn_words.update(text[a:b].lower() for a, b in gcues)
        if not gold_has:
            continue
        in_cue = lambda s, e: any(a <= s < b for a, b in gcues)
        in_scope = lambda s, e: any(a <= s < b for a, b in gscopes)
        for (s, e, par, c) in words:
            if c or in_cue(s, e):
                continue
            g = in_scope(s, e)
            sc_tp += par == 1 and g
            sc_fp += par == 1 and not g
            sc_fn += par == 0 and g
    return {"cue": prf(cue_tp, cue_fp, cue_fn), "scope": prf(sc_tp, sc_fp, sc_fn),
            "n_sentences": len(sentences), "n_gold_negated": sum(bool(c) for _, c, _ in sentences),
            "top_false_positive_cues": fp_words.most_common(10),
            "top_missed_cues": fn_words.most_common(10)}


def load(zbytes):
    out = {}
    with zipfile.ZipFile(io.BytesIO(zbytes)) as z:
        for name in z.namelist():
            if not name.endswith(".xml"):
                continue
            part = ("abstracts" if "abstract" in name.lower() else
                    "full_papers" if "paper" in name.lower() else
                    "clinical" if "clinical" in name.lower() else os.path.basename(name))
            root = ET.fromstring(z.read(name))
            out[part] = [flatten(s) for s in root.iter("sentence")]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=None)
    ap.add_argument("--out", default="scope_eval.json")
    args = ap.parse_args()
    if args.zip:
        data = open(args.zip, "rb").read()
    else:
        data = None
        for url in URLS:
            try:
                print(f"downloading {url} ...")
                data = urllib.request.urlopen(url, timeout=60).read()
                if data[:2] == b"PK":
                    break
                print("   not a zip file, trying next source")
                data = None
            except Exception as e:
                print(f"   failed: {e}")
        if data is None:
            sys.exit("Could not download BioScope. Download bioscope.zip in a browser from "
                     "https://rgai.inf.u-szeged.hu/node/105, add it to the notebook as a dataset, "
                     "and pass --zip /kaggle/input/<dataset>/bioscope.zip")
    corpora = load(data)
    T.use_exclusion_cues(True)          # the lexicon the models were trained with
    detectors = {"training detector": S.word_parities,
                 "extended detector": T.word_parities_ext}
    report = {}
    for part, sents in corpora.items():
        for dname, det in detectors.items():
            r = evaluate(sents, det)
            report[f"{part} | {dname}"] = r
            cp, cr, cf = r["cue"]
            sp, sr, sf = r["scope"]
            print(f"{part:<12} {dname:<18} sentences {r['n_sentences']:>6} "
                  f"(negated {r['n_gold_negated']:>5}) | cue P {cp:.3f} R {cr:.3f} F1 {cf:.3f} "
                  f"| scope P {sp:.3f} R {sr:.3f} F1 {sf:.3f}")
        print(f"   most common false-positive cues: "
              f"{report[f'{part} | training detector']['top_false_positive_cues'][:6]}")
        print(f"   most common missed cues:         "
              f"{report[f'{part} | training detector']['top_missed_cues'][:6]}")
    json.dump(report, open(args.out, "w"), indent=1)
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
