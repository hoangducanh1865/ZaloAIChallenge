"""Agent loop of one sample: model turn -> tool calls on the sample's ToolEnv -> results back to the model, until the model
returns no call (or max_steps). Every call goes through the env, so env.get_tool_calls() is the submission."""

import json
import time

from .prompt import build_messages, results_text, text


def parse_calls(reply):
    """Tool calls of the first JSON object in the reply; None when there is none (the sample then ends)."""
    start = reply.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(reply[start:])
    except json.JSONDecodeError:
        return None
    calls = obj.get("tool_calls") if isinstance(obj, dict) else None
    if not isinstance(calls, list):
        return None
    return [c for c in calls if isinstance(c, dict) and isinstance(c.get("name"), str)]


def run_sample(model, env, sample, catalog, split_dir, max_steps, gen_kwargs, transcript=False):
    t0 = time.time()
    msgs = build_messages(sample, catalog, split_dir, transcript)
    replies = []
    for _ in range(max_steps):
        reply = model.generate(msgs, **gen_kwargs)
        replies.append(reply)
        calls = parse_calls(reply)
        if not calls:
            break
        results = [env.call(c["name"], c.get("arguments") or {}) for c in calls]
        msgs += [text("assistant", reply.strip()), text("user", results_text(results))]
        if any(c["name"] == "sys.ask_missing_info" for c in calls):
            break  # asking the user ends the turn
    return {
        "id": sample["id"],
        "tool_calls": env.get_tool_calls(),
        "state_hash": env.get_state_hash(),
        "replies": replies,
        "seconds": round(time.time() - t0, 2),
    }
