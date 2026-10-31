"""IBM Granite Speech (granite-4.0-1b-speech, granite-speech-4.1-2b, granite-speech-5.0-2b) through transformers: a
speech encoder and projector in front of a Granite LLM. Its chat template takes text only: each audio becomes an
<|audio|> placeholder in the text, and the waveforms go to the processor in the same order."""

import librosa
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


class GraniteSpeech:
    def __init__(self, model_id, dtype="bfloat16"):
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, dtype=getattr(torch, dtype),
                                                               device_map="cuda").eval()
        self.prompt_tokens = 0

    @torch.inference_mode()
    def generate(self, messages, max_new_tokens=512, enable_thinking=False):
        chat, wavs = [], []
        for m in messages:
            parts = []
            for c in m["content"]:
                if c["type"] == "text":
                    parts.append(c["text"])
                else:
                    parts.append("<|audio|>")
                    wavs.append(librosa.load(c["audio"], sr=16000)[0])
            chat.append({"role": m["role"], "content": "\n".join(parts)})
        prompt = self.processor.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(prompt, wavs, device=self.model.device, return_tensors="pt").to(self.model.device)
        self.prompt_tokens = inputs["input_ids"].shape[-1]
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1)
        return self.processor.tokenizer.decode(out[0][self.prompt_tokens:], skip_special_tokens=True).strip()
