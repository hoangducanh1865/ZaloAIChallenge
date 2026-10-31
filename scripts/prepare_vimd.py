"""
Small ASR set for stage 1 from ViMD (Vietnamese Multi-Dialect, 63 provinces; HF nguyendv02/ViMD_Dataset, CC-BY-NC-ND:
check the licence before any use in the final solution): N train shards spread over the provinces (shards are sorted
by province), each utterance written as a 16 kHz wav plus one manifest line {"audio", "text", "region", "province"}.

Usage: python scripts/prepare_vimd.py <out_dir> [--shards 15] [--max_sec 20]

Author: Duc Anh Hoang
"""

import argparse
import io
import json
from pathlib import Path

import librosa
import pyarrow.parquet as pq
import requests
import soundfile as sf

URL = "https://huggingface.co/datasets/nguyendv02/ViMD_Dataset/resolve/main/data/train-{:05d}-of-00103.parquet"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--shards", type=int, default=15)
    ap.add_argument("--max_sec", type=float, default=20.0)
    a = ap.parse_args()
    (a.out_dir / "wav").mkdir(parents=True, exist_ok=True)
    with open(a.out_dir / "manifest.jsonl", "w", encoding="utf-8") as man:
        for k in range(a.shards):
            i = round(k * 102 / max(a.shards - 1, 1))
            shard = a.out_dir / f"shard_{i:05d}.parquet"
            if not shard.exists():
                shard.write_bytes(requests.get(URL.format(i), timeout=600).content)
            kept = 0
            for r in pq.read_table(shard).to_pylist():
                wav, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
                if wav.ndim > 1:
                    wav = wav.mean(1)
                if len(wav) / sr > a.max_sec or not r["text"].strip():
                    continue
                if sr != 16000:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
                path = a.out_dir / "wav" / r["filename"]
                sf.write(path, wav, 16000)
                man.write(json.dumps({"audio": str(path), "text": r["text"].strip(), "region": r["region"],
                                      "province": r["province_name"]}, ensure_ascii=False) + "\n")
                kept += 1
            shard.unlink()  # the wavs are kept; the parquet is not needed again
            print(f"shard {i}: kept {kept}", flush=True)


if __name__ == "__main__":
    main()
