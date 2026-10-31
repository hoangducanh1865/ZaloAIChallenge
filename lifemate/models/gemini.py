"""Gemini through an OpenAI-compatible endpoint (default: the local router http://localhost:20128/v1, models
"ag/gemini-..."; GEMINI_API_KEY and GEMINI_BASE_URL from the environment or .env), audio sent inline as base64 wav.
A reference point only: an external API cannot be used in the competition (offline, < 6B parameters)."""

import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv


class Gemini:
    def __init__(self, model_id, base_url=None):
        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
        self.url = (base_url or os.environ.get("GEMINI_BASE_URL", "http://localhost:20128/v1")) + "/chat/completions"
        self.key, self.model_id = os.environ["GEMINI_API_KEY"], model_id
        self.prompt_tokens = 0

    def generate(self, messages, max_new_tokens=512, enable_thinking=False):  # thinking level is in the model name
        msgs = [{"role": m["role"], "content": [
            {"type": "text", "text": c["text"]} if c["type"] == "text" else
            {"type": "input_audio", "input_audio": {"data": base64.b64encode(Path(c["audio"]).read_bytes()).decode(),
                                                    "format": "wav"}}
            for c in m["content"]]} for m in messages]
        body = json.dumps({"model": self.model_id, "messages": msgs, "temperature": 0, "stream": False}).encode()
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"}
        for i in range(5):  # rate limits and router hiccups
            try:
                r = json.load(urllib.request.urlopen(urllib.request.Request(self.url, body, headers), timeout=300))
                content = r["choices"][0]["message"]["content"]
                break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError):  # stream / error bodies
                if i == 4:
                    raise
                time.sleep(2 ** i * 5)
        self.prompt_tokens = (r.get("usage") or {}).get("prompt_tokens", 0)
        return content or ""
