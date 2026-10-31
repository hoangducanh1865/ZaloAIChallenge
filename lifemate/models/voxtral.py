"""Mistral Voxtral (Mini 3B: Whisper encoder + Ministral 3B) through transformers. Its template has no system role and
wants alternating user / assistant turns: the system prompt becomes user text, consecutive turns of a role are merged."""

import torch
from transformers import AutoProcessor, VoxtralForConditionalGeneration


class Voxtral:
    def __init__(self, model_id, dtype="bfloat16"):
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = VoxtralForConditionalGeneration.from_pretrained(model_id, dtype=getattr(torch, dtype),
                                                                     device_map="cuda").eval()
        self.prompt_tokens = 0

    @torch.inference_mode()
    def generate(self, messages, max_new_tokens=512, enable_thinking=False):
        conv = []
        for m in messages:
            role = "assistant" if m["role"] == "assistant" else "user"
            content = [{"type": "text", "text": c["text"]} if c["type"] == "text" else {"type": "audio", "path": c["audio"]}
                       for c in m["content"]]
            if conv and conv[-1]["role"] == role:
                conv[-1]["content"] += content
            else:
                conv.append({"role": role, "content": content})
        for turn in conv:  # assistant turns are plain text
            if turn["role"] == "assistant":
                turn["content"] = "\n".join(c["text"] for c in turn["content"])
        inputs = self.processor.apply_chat_template(conv, return_tensors="pt").to(self.model.device, self.model.dtype)
        self.prompt_tokens = inputs["input_ids"].shape[-1]
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.batch_decode(out[:, self.prompt_tokens:], skip_special_tokens=True)[0].strip()
