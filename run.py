"""
Run an audio LM agent on a split of LifeMate: <out_dir>/preds.jsonl gets one line per sample as soon as it is done (a
re-run resumes after the last saved sample), then submission.jsonl and, on a labelled split (train), score.json.

Usage (from the repo root):
    python run.py model=gemma4_e4b data=public_test
    python run.py model=gemma4_e4b data=train data.limit=200
    python run.py ... data.shard=0 data.num_shards=2   # two processes on one GPU; the last one to finish merges

Author: Duc Anh Hoang
"""

import json
import logging
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import instantiate

from lifemate.agent import run_sample
from lifemate.data import env_factory, load_catalog, load_samples
from lifemate.score import score_run

log = logging.getLogger(__name__)


def read_preds(out):
    preds = {}
    for p in sorted(out.glob("preds*.jsonl")):
        with open(p, encoding="utf-8") as f:
            preds.update((r["id"], r) for r in map(json.loads, f) if r)
    return preds


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg):
    random.seed(0), np.random.seed(0), torch.manual_seed(0)  # decoding is greedy; seeds for the competition rule
    d, out = cfg.data, Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    split_dir = Path(cfg.data_root) / d.split
    samples = load_samples(cfg.data_root, d.split, d.limit)
    done = read_preds(out)
    todo = [s for s in samples[d.shard :: d.num_shards] if s["id"] not in done]
    log.info(f"{cfg.run_name}: {len(samples)} samples, {len(done)} done, {len(todo)} to run in shard {d.shard}")

    if todo:
        model = instantiate(cfg.model)
        catalog, make_env = load_catalog(cfg.data_root, d.split), env_factory(cfg.data_root, d.split)
        gen = {"max_new_tokens": cfg.agent.max_new_tokens, "enable_thinking": cfg.agent.enable_thinking}
        name = "preds.jsonl" if d.num_shards == 1 else f"preds_{d.shard}of{d.num_shards}.jsonl"
        with open(out / name, "a", encoding="utf-8") as f:
            for i, s in enumerate(todo, 1):
                env = make_env(s["id"])  # one ToolEnv per sample, created once
                try:
                    r = run_sample(model, env, s, catalog, split_dir, cfg.agent.max_steps, gen)
                except Exception as e:  # e.g. CUDA OOM on a long history: keep the calls already made
                    log.exception(s["id"])
                    r = {"id": s["id"], "tool_calls": env.get_tool_calls(), "state_hash": env.get_state_hash(),
                         "error": repr(e)}
                r["prompt_tokens"] = model.prompt_tokens
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                log.info(f"[{i}/{len(todo)}] {s['id']} calls={len(r['tool_calls'])} {r.get('seconds')}s "
                         f"tokens={r['prompt_tokens']}")

    preds = read_preds(out)
    if any(s["id"] not in preds for s in samples):
        return  # other shards still running
    with open(out / "submission.jsonl", "w", encoding="utf-8") as f:
        for s in samples:
            p = preds[s["id"]]
            f.write(json.dumps({k: p[k] for k in ("id", "tool_calls", "state_hash")}, ensure_ascii=False) + "\n")
    log.info(f"wrote {out / 'submission.jsonl'}")
    sec = sorted(preds[s["id"]].get("seconds", 0) for s in samples)  # wall time per sample: the 30 s limit is per sample
    lat = {"n": len(sec), "mean": sum(sec) / len(sec), "p50": sec[len(sec) // 2], "p95": sec[int(0.95 * (len(sec) - 1))],
           "max": sec[-1], "over_30s": sum(x > 30 for x in sec)}
    json.dump(lat, open(out / "latency.json", "w"), indent=1)
    log.info(f"LATENCY {json.dumps(lat)}")
    if "label" in samples[0]:
        r = score_run(cfg.data_root, d.split, [preds[s["id"]] for s in samples])
        json.dump(r, open(out / "score.json", "w"), indent=1)
        log.info(f"SCORE {d.split} {r['score']:.2f} on {r['n']} samples")


if __name__ == "__main__":
    main()
