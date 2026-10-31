"""
Fine-tune Gemma 4 (E2B / E4B) with LoRA on the LLM plus trainable audio projector (and optionally the top audio
encoder layers). Loss only on the model turns of each conversation.

    stage=asr    stage 1, listening: <ASR prompt + audio> -> transcript, from a manifest (scripts/prepare_vimd.py)
    stage=agent  stage 2, tool use: the LifeMate train samples (minus the first holdout ones, kept for evaluation),
                 one gold call per model turn with its gold result, then {"tool_calls": []}

Usage (from the repo root):
    python train.py stage=asr
    python train.py stage=agent init_adapter=<stage 1 out_dir>/adapter
An adapter (LoRA + projector) is saved to <out_dir>/adapter every save_every steps and at the end; a re-run resumes
from it. Evaluate with run.py / scripts/asr_probe.py and +model.adapter=<out_dir>/adapter.

Author: Duc Anh Hoang
"""

import json
import logging
import math
import random
from pathlib import Path

import hydra
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForMultimodalLM, AutoProcessor

from lifemate.data import load_catalog, load_samples
from lifemate.prompt import build_messages, calls_json, results_text, text

log = logging.getLogger(__name__)
ASR_PROMPT = "Chép lại nguyên văn lời nói tiếng Việt trong audio, chỉ trả về câu chép lại, không giải thích."


def asr_conversations(cfg):
    with open(cfg.stage.manifest, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    random.Random(0).shuffle(rows)
    rows = rows[: cfg.stage.limit] if cfg.stage.limit else rows
    return [[{"role": "user", "content": [{"type": "text", "text": ASR_PROMPT}, {"type": "audio", "audio": r["audio"]}]},
             text("assistant", r["text"])] for r in rows]


def agent_conversations(cfg, part="train"):
    """train[:holdout] = test (kept for run.py), the next `val` samples = validation, the rest = training."""
    split = "train"
    catalog, split_dir = load_catalog(cfg.data_root, split), Path(cfg.data_root) / split
    h, v = cfg.stage.holdout, cfg.stage.val
    samples = load_samples(cfg.data_root, split)
    samples = samples[h : h + v] if part == "val" else samples[h + v :]
    convs = []
    for s in samples:
        msgs = build_messages(s, catalog, split_dir, rules=cfg.stage.rules)
        calls, results = s["label"]["tool_calls"], s["label"]["tool_results"]
        for c, r in zip(calls, results):
            msgs += [text("assistant", calls_json([c])), text("user", results_text([r]))]
            if c["name"] == "sys.ask_missing_info":
                msgs.pop()  # asking ends the turn (as in lifemate/agent.py): no result, no closing turn
                break
        else:
            msgs.append(text("assistant", calls_json([])))
        convs.append(msgs)
    if cfg.stage.limit and part == "train":  # a fixed random subset, for short experiments
        convs = random.Random(0).sample(convs, min(cfg.stage.limit, len(convs)))
    return convs


class Encoder:
    """Chat-template a conversation and mask everything but the model turns: '<|turn>model\\n' ... '<turn|>'."""

    def __init__(self, processor, max_tokens):
        self.p, self.max_tokens = processor, max_tokens
        tok = processor.tokenizer
        self.start = tok.encode("<|turn>model\n", add_special_tokens=False)
        self.end = tok.convert_tokens_to_ids("<turn|>")

    def __call__(self, msgs):
        x = self.p.apply_chat_template(msgs, tokenize=True, return_dict=True, return_tensors="pt")
        ids = x["input_ids"][0].tolist()
        if len(ids) > self.max_tokens:
            return None
        labels, n, i = [-100] * len(ids), len(self.start), 0
        while i < len(ids):
            if ids[i : i + n] == self.start:
                j = i + n
                while j < len(ids) and ids[j] != self.end:
                    labels[j] = ids[j]
                    j += 1
                if j < len(ids):
                    labels[j] = ids[j]  # learn to close the turn
                i = j
            i += 1
        x["labels"] = torch.tensor([labels])
        return x


def target_loss(model, x):
    """Cross-entropy on the labelled tokens only: logits for those positions alone (logits_to_keep), instead of the
    full sequence x 262K vocabulary that model(labels=...) materialises."""
    labels = x.pop("labels")[0]
    pos = (labels[1:] != -100).nonzero().squeeze(1)  # position t predicts token t + 1
    logits = model(**x, logits_to_keep=pos).logits[0].float()
    return torch.nn.functional.cross_entropy(logits, labels[pos + 1])


@torch.no_grad()
def val_loss(model, enc, convs):
    model.eval()
    losses = [target_loss(model, x.to(model.device)).item() for x in map(enc, convs) if x is not None]
    model.train()
    return sum(losses) / max(len(losses), 1)


def build_model(cfg):
    model = AutoModelForMultimodalLM.from_pretrained(cfg.model.model_id, dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    out_adapter = Path(cfg.out_dir) / "adapter"
    start = out_adapter if (out_adapter / "adapter_config.json").exists() else (Path(cfg.init_adapter) if cfg.init_adapter else None)
    if start:
        log.info(f"adapter from {start}")
        model = PeftModel.from_pretrained(model, str(start), is_trainable=True)
    else:
        top = [f"model.audio_tower.layers.{i}" for i in cfg.stage.train_audio_layers]
        model = get_peft_model(model, LoraConfig(
            r=cfg.lora.r, lora_alpha=cfg.lora.alpha, lora_dropout=0.05,
            target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
            modules_to_save=["embed_audio", "audio_tower.output_proj", *top]))
    model.print_trainable_parameters()
    return model


@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg):
    torch.manual_seed(0)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    convs = asr_conversations(cfg) if cfg.stage.name == "asr" else agent_conversations(cfg)
    val = agent_conversations(cfg, "val") if cfg.stage.name == "agent" and cfg.stage.val else []
    processor = AutoProcessor.from_pretrained(cfg.model.model_id)
    enc = Encoder(processor, cfg.max_tokens)
    model = build_model(cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.stage.lr, weight_decay=0.0)
    total = math.ceil(len(convs) * cfg.stage.epochs / cfg.grad_accum)
    state_path = out / "state.json"
    step = json.load(open(state_path))["step"] if state_path.exists() else 0
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / cfg.warmup) * max(0.05, 1 - s / max(total, 1)), last_epoch=-1)
    for _ in range(step):
        sched.step()
    log.info(f"{cfg.stage.name}: {len(convs)} conversations (val {len(val)}), {cfg.stage.epochs} epochs, {total} steps, "
             f"resume at {step}")
    per_epoch = math.ceil(len(convs) / cfg.grad_accum)
    best = json.load(open(state_path)).get("best", float("inf")) if state_path.exists() else float("inf")

    order = [i for e in range(cfg.stage.epochs) for i in random.Random(e).sample(range(len(convs)), len(convs))]
    model.train()
    running, seen, skipped = 0.0, 0, 0
    for k in range(step * cfg.grad_accum, len(order)):
        x = enc(convs[order[k]])
        if x is None:
            skipped += 1
            continue
        loss = target_loss(model, x.to(model.device)) / cfg.grad_accum
        loss.backward()
        running += loss.item()
        if (k + 1) % cfg.grad_accum:
            continue
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(), sched.step(), opt.zero_grad(set_to_none=True)
        step += 1
        seen += 1
        if step % cfg.log_every == 0:
            log.info(f"step {step}/{total} loss {running / seen:.4f} lr {sched.get_last_lr()[0]:.2e} skipped {skipped}")
            running, seen = 0.0, 0
        if val and (step % per_epoch == 0 or step == total):  # end of an epoch: keep the best adapter on val
            v = val_loss(model, enc, val)
            log.info(f"VAL step {step} loss {v:.4f} (best {best:.4f})")
            if v < best:
                best = v
                model.save_pretrained(out / "adapter_best")
        if step % cfg.save_every == 0 or step == total:
            model.save_pretrained(out / "adapter")
            json.dump({"step": step, "best": best}, open(state_path, "w"))
    model.save_pretrained(out / "adapter")
    json.dump({"step": step, "best": best, "done": True}, open(state_path, "w"))
    log.info(f"TRAIN DONE {cfg.stage.name} step {step}")


if __name__ == "__main__":
    main()
