"""Text-only LLM through transformers, for the transcript diagnostic (agent.transcript=true): measures the tool-use
ability of a candidate LLM backbone for an encoder + projector + LLM model built in-house. Rejects audio input."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class TextLM:
    def __init__(self, model_id, dtype="bfloat16"):
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype), device_map="cuda").eval()
        self.prompt_tokens = 0

    @torch.inference_mode()
    def generate(self, messages, max_new_tokens=512, enable_thinking=False):
        chat = [{"role": m["role"], "content": "\n".join(c["text"] for c in m["content"])} for m in messages]
        ids = self.tok.apply_chat_template(chat, add_generation_prompt=True, return_tensors="pt",
                                           enable_thinking=enable_thinking).to(self.model.device)
        self.prompt_tokens = ids.shape[-1]
        out = self.model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tok.decode(out[0][self.prompt_tokens:], skip_special_tokens=True).strip()
