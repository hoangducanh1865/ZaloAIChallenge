# ZaloAIChallenge — LifeMate

Zalo AI Challenge 2026, LifeMate: a spoken Vietnamese request -> tool calls on simulated apps (`de_bai.txt`).
Approach: an end-to-end audio LM agent (no separate ASR). The model sees the system prompt (rules, user context, schemas
of the installed tools), the earlier turns and the request audio, and answers each turn with
`{"tool_calls": [...]}`; the calls run on the organisers' `zaic_tools` and their results go back to the model until it
returns no call.

| Path | Content |
|---|---|
| `run.py` | Hydra entry point: one model on one split -> `preds.jsonl`, `submission.jsonl`, `score.json` (train) |
| `configs/model/*.yaml` | one file per model (`_target_` = adapter class); `configs/data/*.yaml`: split, limit, shards |
| `lifemate/models/` | adapters with `generate(messages) -> str`: `gemma.py` (Gemma 4), `qwen_omni.py` (Qwen2.5-Omni), `granite_speech.py`, `voxtral.py`, `kimi_audio.py` |
| `lifemate/prompt.py`, `agent.py` | messages of a sample; agent loop on one `ToolEnv` |
| `lifemate/score.py` | local metric (state hash + tool-call F1), for train |
| `scripts/` | `colab_ssh.sh`, `a100_eval.sh` (A100 job), `fetch_drive_folder.py` (data), `setup_kimi_audio.sh` |
| `notebooks/colab.ipynb` | mount Drive, clone/pull, open SSH |

```bash
python run.py model=gemma4_e4b data=public_test                 # leaderboard file
python run.py model=gemma4_e4b data=train data.limit=200        # local score
python -m lifemate.score --root <data_root> --split train --preds <run>/preds.jsonl
```

Adding a model: `lifemate/models/<name>.py` with `generate(messages, max_new_tokens, enable_thinking) -> str` and
`prompt_tokens`, plus `configs/model/<name>.yaml`. Competition limits: LLM < 5B parameters, whole system < 6B,
context <= 16K tokens, <= 30 s per sample on an A30, greedy decoding.
