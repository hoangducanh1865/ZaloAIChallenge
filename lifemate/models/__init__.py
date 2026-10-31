"""One adapter per model family, all with generate(messages, max_new_tokens, enable_thinking) -> str, where messages
is the format of lifemate/prompt.py. A new model = one file here + one configs/model/<name>.yaml (_target_ = its class)."""
