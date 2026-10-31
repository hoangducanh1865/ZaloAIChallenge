"""Local re-implementation of the LifeMate metric (CHALLENGE_INFO.md, Evaluation) for labelled splits (train).

Usage: python -m lifemate.score --root <data_root> --split train --preds <run>/preds.jsonl

Gold state_hash = the gold label.tool_calls replayed on a fresh ToolEnv, as the organisers replay a submission.
A call matches a gold call when the name is the same, every non-fuzzy argument is identical and every fuzzy argument
(x-match: fuzzy) has token-F1 > 0.6 after lower-casing and NFC; each gold call matches at most one call.
"""

import argparse
import json
import unicodedata
from collections import Counter

from .data import env_factory, load_catalog, load_samples


def token_f1(a, b):
    ta, tb = (unicodedata.normalize("NFC", str(x)).lower().split() for x in (a, b))
    common = sum((Counter(ta) & Counter(tb)).values())
    if not ta or not tb or not common:
        return float(ta == tb)
    p, r = common / len(ta), common / len(tb)
    return 2 * p * r / (p + r)


def fuzzy_params(catalog):
    return {n: {k for k, p in t["parameters"].get("properties", {}).items() if p.get("x-match") == "fuzzy"}
            for n, t in catalog.items()}


def call_match(pred, gold, fuzzy):
    if pred["name"] != gold["name"]:
        return False
    pa, ga, fz = pred.get("arguments") or {}, gold.get("arguments") or {}, fuzzy.get(gold["name"], set())
    for k in set(pa) | set(ga):
        if k in fz:
            if token_f1(pa.get(k, ""), ga.get(k, "")) <= 0.6:
                return False
        elif pa.get(k) != ga.get(k):
            return False
    return True


def sample_score(pred_calls, pred_hash, gold_calls, gold_hash, fuzzy):
    if not gold_calls:
        return float(not pred_calls)
    if pred_hash != gold_hash:
        return 0.0
    free, m = list(gold_calls), 0
    for p in pred_calls:
        j = next((i for i, g in enumerate(free) if call_match(p, g, fuzzy)), None)
        if j is not None:
            free.pop(j)
            m += 1
    if m == 0:
        return 0.0
    prec, rec = m / len(pred_calls), m / len(gold_calls)
    return 2 * prec * rec / (prec + rec)


def score_run(root, split, preds):
    """{"score": 0-100, "n": ..., "per_sample": {id: score}} over the samples in preds."""
    gold = {s["id"]: s for s in load_samples(root, split)}
    fuzzy, make_env = fuzzy_params(load_catalog(root, split)), env_factory(root, split)
    per = {}
    for p in preds:
        g = gold[p["id"]]["label"]["tool_calls"]
        env = make_env(p["id"])
        for c in g:
            env.call(c["name"], c["arguments"])
        per[p["id"]] = sample_score(p["tool_calls"], p["state_hash"], g, env.get_state_hash(), fuzzy)
    return {"score": 100 * sum(per.values()) / max(len(per), 1), "n": len(per), "per_sample": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--preds", required=True)
    a = ap.parse_args()
    with open(a.preds, encoding="utf-8") as f:
        preds = [json.loads(line) for line in f if line.strip()]
    r = score_run(a.root, a.split, preds)
    print(f"{a.split}: {r['score']:.2f} on {r['n']} samples")


if __name__ == "__main__":
    main()
