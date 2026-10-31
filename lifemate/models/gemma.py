"""Gemma 4 (E2B / E4B / 12B take audio) through transformers, greedy decoding."""

import re

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

SPECIAL = re.compile(r"<\|[^>]*>|<[^>]*\|>")  # control tokens such as <|turn>, <turn|>, <channel|>


class Gemma:
    def __init__(self, model_id, dtype="bfloat16"):
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForMultimodalLM.from_pretrained(model_id, dtype=getattr(torch, dtype), device_map="cuda")
        self.model.eval()
        self.prompt_tokens = 0  # of the last call, to check the 16K context limit of the competition

    @torch.inference_mode()
    def generate(self, messages, max_new_tokens=512, enable_thinking=False):
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True,
            enable_thinking=enable_thinking,
        ).to(self.model.device)
        self.prompt_tokens = inputs["input_ids"].shape[-1]
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        raw = self.processor.decode(out[0][self.prompt_tokens:], skip_special_tokens=False)
        return SPECIAL.sub("", raw.split("<channel|>")[-1]).strip()  # drop the thought block when thinking is on
