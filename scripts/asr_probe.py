"""
How well an audio LM hears Vietnamese commands: transcribe the last request of the first N train samples with the
model's own adapter and compare with audio_text (WER and CER, lower-cased, punctuation removed).

Usage (from the repo root):
    python scripts/asr_probe.py model=gemma4_e2b data.limit=100 out_dir=<dir>   # Hydra overrides of configs/config.yaml
Writes <out_dir>/asr.jsonl (one line per sample, appended as it goes; a re-run resumes) and asr.json (mean WER, CER).

Author: Duc Anh Hoang
"""

import json
import re
import sys
import unicodedata
from pathlib import Path

import hydra
from hydra.utils import instantiate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lifemate.data import load_samples  # noqa: E402

PROMPT = "Chép lại nguyên văn lời nói tiếng Việt trong audio, chỉ trả về câu chép lại, không giải thích."


def norm(s):
    s = unicodedata.normalize("NFC", s.lower())
    return re.sub(r"[^\w\s]", " ", s).split()


def edit(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[-1]


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg):
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "asr.jsonl"
    done = {json.loads(line)["id"] for line in open(path)} if path.exists() else set()
    samples = [s for s in load_samples(cfg.data_root, cfg.data.split, cfg.data.limit) if s["id"] not in done]
    if samples:
        model = instantiate(cfg.model)
        with open(path, "a", encoding="utf-8") as f:
            for s in samples:
                wav = str(Path(cfg.data_root) / cfg.data.split / s["audio"])
                msgs = [{"role": "user", "content": [{"type": "text", "text": PROMPT}, {"type": "audio", "audio": wav}]}]
                hyp = model.generate(msgs, max_new_tokens=200)
                ref, h = norm(s["audio_text"]), norm(hyp)
                cref, ch = " ".join(ref), " ".join(h)
                r = {"id": s["id"], "ref": s["audio_text"], "hyp": hyp, "wer": edit(ref, h) / max(len(ref), 1),
                     "cer": edit(cref, ch) / max(len(cref), 1)}
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
    rows = [json.loads(line) for line in open(path)]
    summary = {k: sum(r[k] for r in rows) / len(rows) for k in ("wer", "cer")} | {"n": len(rows)}
    json.dump(summary, open(out / "asr.json", "w"), indent=1)
    print(f"ASR {cfg.run_name}: {json.dumps(summary)}")


if __name__ == "__main__":
    main()
