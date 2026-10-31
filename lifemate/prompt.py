"""Chat messages of one sample, in a model-neutral format that every adapter of lifemate/models converts:
[{"role": "system" | "user" | "assistant", "content": [{"type": "text", "text": ...} | {"type": "audio", "audio": <wav path>}]}]

The model answers every turn with one JSON object {"tool_calls": [{"name": ..., "arguments": {...}}, ...]}; the agent runs
the calls and sends their results back as a user turn; {"tool_calls": []} ends the sample.
"""

import json
from pathlib import Path

RULES = """Bạn là trợ lý giọng nói tiếng Việt điều khiển các app trên thiết bị của người dùng bằng cách gọi tool.
Mỗi lượt, chỉ trả lời đúng một JSON trên một dòng, không viết gì khác:
{"tool_calls": [{"name": "<app>.<tool>", "arguments": {...}}]}
Hệ thống chạy các tool đó và gửi lại kết quả; khi đã làm xong yêu cầu thì trả {"tool_calls": []}.

Quy tắc:
- Chỉ xử lý yêu cầu cuối cùng (audio cuối); các lượt trước là ngữ cảnh, việc của chúng đã được làm.
- ID (thiết bị, liên hệ, báo thức, email...) phải lấy từ kết quả tool đọc (list/search/get_status), không tự bịa. Nếu cần ID,
  gọi tool đọc trước rồi chờ kết quả ở lượt sau.
- Chỉ gọi tool cần thiết; không gọi lại tool đã có kết quả; tham số đúng tên, kiểu và enum trong schema.
- Thời gian: ISO 8601 có múi giờ, ví dụ "2026-11-11T05:30:00+07:00", tính theo current_time.
- Thiếu tham số bắt buộc mà không suy ra được từ lời nói, ngữ cảnh hay kết quả tool: gọi
  {"name": "sys.ask_missing_info", "arguments": {"tool_name": "<tool cần gọi>", "params": ["<tham số thiếu>"]}} rồi kết thúc.
- Trả ngay {"tool_calls": []} mà không gọi tool nào khi: không app nào đã cài làm được việc này, người dùng chỉ trò chuyện,
  cảm ơn, đổi ý hay huỷ yêu cầu, hoặc giá trị yêu cầu không hợp lệ.
- Yêu cầu có điều kiện ("nếu ... thì ..."): đọc dữ liệu để kiểm tra, chỉ làm khi điều kiện đúng."""


def _tool_line(spec):
    keep = {k: spec[k] for k in ("name", "kind", "description", "parameters", "returns") if spec.get(k)}
    return json.dumps(keep, ensure_ascii=False, separators=(",", ":"))


def system_prompt(sample, catalog):
    ctx = {k: v for k, v in sample["context"].items() if k != "installed_apps"}
    names = [t for app in sample["context"]["installed_apps"].values() for t in app["tools"]] + ["sys.ask_missing_info"]
    tools = "\n".join(_tool_line(catalog[n]) for n in names if n in catalog)
    return f"{RULES}\n\nNgữ cảnh người dùng:\n{json.dumps(ctx, ensure_ascii=False)}\n\nCác tool được phép gọi:\n{tools}"


def text(role, s):
    return {"role": role, "content": [{"type": "text", "text": s}]}


def calls_json(calls, reply=None):
    out = {"tool_calls": [{"name": c["name"], "arguments": c["arguments"]} for c in calls]}
    if reply:
        out["reply"] = reply
    return json.dumps(out, ensure_ascii=False)


def results_text(results):
    return "Kết quả tool: " + json.dumps(results, ensure_ascii=False, separators=(",", ":"))


def audio_turn(label, wav):
    return {"role": "user", "content": [{"type": "text", "text": label}, {"type": "audio", "audio": str(wav)}]}


def build_messages(sample, catalog, split_dir):
    """System prompt, the earlier turns (user audio, assistant calls / results / reply) and the last request as audio."""
    split_dir = Path(split_dir)
    msgs = [text("system", system_prompt(sample, catalog))]
    for turn in sample["history"]:
        if turn["role"] == "user":
            msgs.append(audio_turn("Lượt trước của người dùng:", split_dir / turn["audio"]))
        else:
            if turn.get("tool_calls"):
                msgs.append(text("assistant", calls_json(turn["tool_calls"])))
                msgs.append(text("user", results_text(turn.get("tool_results", []))))
            msgs.append(text("assistant", calls_json([], turn.get("text"))))
    msgs.append(audio_turn("Yêu cầu cuối của người dùng (cần xử lý):", split_dir / sample["audio"]))
    return msgs
