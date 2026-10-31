"""Kimi-Audio through its own inference code (kimia_infer, github.com/MoonshotAI/Kimi-Audio), text output only."""

from kimia_infer.api.kimia import KimiAudio as _KimiAudio


class KimiAudio:
    def __init__(self, model_id):
        self.model = _KimiAudio(model_path=model_id, load_detokenizer=False)
        self.prompt_tokens = 0

    def generate(self, messages, max_new_tokens=512, enable_thinking=False):
        # Kimi-Audio takes a flat list of single-modality messages and has no system role: the system prompt is a user text.
        conv = [
            {"role": "assistant" if m["role"] == "assistant" else "user",
             "message_type": c["type"], "content": c["text"] if c["type"] == "text" else c["audio"]}
            for m in messages for c in m["content"]
        ]
        _, text = self.model.generate(conv, output_type="text", text_temperature=0.0, text_top_k=1,
                                      max_new_tokens=max_new_tokens)
        return text
