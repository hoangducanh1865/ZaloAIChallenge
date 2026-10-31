"""Qwen2.5-Omni through transformers, text output only: the Talker (speech output) is dropped after loading, so only the
Thinker runs (audio encoder + LLM; Qwen2.5-Omni-3B: 3B LLM, within the competition limits)."""

import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor


class QwenOmni:
    def __init__(self, model_id, dtype="bfloat16"):
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_id)
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_id, dtype=getattr(torch, dtype),
                                                                    device_map="cuda")
        model.disable_talker()
        self.model = model.thinker.eval()
        self.prompt_tokens = 0

    @torch.inference_mode()
    def generate(self, messages, max_new_tokens=512, enable_thinking=False):  # no thinking mode in Qwen2.5-Omni
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True,
        ).to(self.model.device).to(self.model.dtype)
        self.prompt_tokens = inputs["input_ids"].shape[-1]
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.decode(out[0][self.prompt_tokens:], skip_special_tokens=True).strip()
