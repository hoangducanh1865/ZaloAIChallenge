"""Samples, tool catalog and the organisers' zaic_tools of one split (each split ships its own zaic_tools version)."""

import json
import sys
from pathlib import Path


def load_samples(root, split, limit=None, shard=0, num_shards=1):
    with open(Path(root) / split / f"{split}.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[:limit][shard::num_shards]


def load_catalog(root, split):
    """{tool name: tool spec} of every app of the split."""
    with open(Path(root) / split / f"{split}_tools_catalog.json", encoding="utf-8") as f:
        return {t["name"]: t for app in json.load(f)["apps"] for t in app["tools"]}


def env_factory(root, split):
    """ToolEnv(question_id) of the split; one per sample, created once (competition rule)."""
    sys.path.insert(0, str(Path(root) / split))
    from zaic_tools import ToolEnv

    return lambda question_id: ToolEnv(split, question_id, root=root)
