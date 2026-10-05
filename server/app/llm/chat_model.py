import json
from collections.abc import Callable
from typing import Any
from urllib.request import Request, urlopen

from app.activity.trace import Cancellation
from app.chat.cancel import TurnCancelled
from app.core.settings import load_settings


class ChatModelFailed(Exception):
    pass


def selected_chat_model(default: str) -> str:
    try:
        from app.agent.context import active_turn
        from app.organizations.models import organization_chat_model

        context = active_turn.get()
        if context is not None and context.organization_id != "":
            return organization_chat_model(context.organization_id)
    except (ImportError, OSError):
        pass
    return default


def generate_json(
    prompt: str,
    schema: dict[str, Any],
    timeout_seconds: int,
    model: str | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    settings = load_settings()
    options: dict[str, Any] = {"temperature": 0, "num_ctx": settings.ollama_num_ctx}
    options["num_predict"] = settings.answer_max_tokens if max_tokens is None else max_tokens
    request = Request(
        f"{settings.ollama_base_url.rstrip('/')}/api/chat",
        data=json.dumps(
            {
                "model": model or selected_chat_model(settings.chat_model),
                "stream": False,
                "think": False,
                "keep_alive": settings.ollama_keep_alive,
                "options": options,
                "messages": [{"role": "user", "content": prompt}],
                "format": schema,
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = json.load(response)
        content = payload.get("message", {}).get("content", "")
        parsed = json.loads(content) if isinstance(content, str) else None
    except (OSError, json.JSONDecodeError) as error:
        raise ChatModelFailed("The chat model did not answer.") from error
    if not isinstance(parsed, dict):
        raise ChatModelFailed("The chat model returned an unreadable answer.")
    return parsed


def partial_answer(buffer: str) -> str:
    marker = '"answer"'
    index = buffer.find(marker)
    if index < 0:
        return ""
    rest = buffer[index + len(marker) :].lstrip()
    if not rest.startswith(":"):
        return ""
    rest = rest[1:].lstrip()
    if not rest.startswith('"'):
        return ""
    return decode_partial_string(rest[1:])


def decode_partial_string(raw: str) -> str:
    characters: list[str] = []
    index = 0
    while index < len(raw):
        character = raw[index]
        if character == '"':
            break
        if character != "\\":
            characters.append(character)
            index += 1
            continue
        if index + 1 >= len(raw):
            break
        escaped = raw[index + 1]
        simple = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "/": "/"}
        if escaped in simple:
            characters.append(simple[escaped])
            index += 2
            continue
        if escaped == "u":
            if index + 6 > len(raw):
                break
            try:
                characters.append(chr(int(raw[index + 2 : index + 6], 16)))
            except ValueError:
                break
            index += 6
            continue
        break
    return "".join(characters)


def stream_json(
    prompt: str,
    schema: dict[str, Any],
    timeout_seconds: int,
    on_answer_delta: Callable[[str], None],
    cancel: Cancellation | None,
    ready_to_show: Callable[[str], bool] | None = None,
) -> dict[str, Any]:
    settings = load_settings()
    request = Request(
        f"{settings.ollama_base_url.rstrip('/')}/api/chat",
        data=json.dumps(
            {
                "model": selected_chat_model(settings.chat_model),
                "stream": True,
                "think": False,
                "keep_alive": settings.ollama_keep_alive,
                "options": {
                    "temperature": 0,
                    "num_ctx": settings.ollama_num_ctx,
                    "num_predict": settings.answer_max_tokens,
                },
                "messages": [{"role": "user", "content": prompt}],
                "format": schema,
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    buffer = ""
    sent = 0
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            if cancel is not None:
                cancel.bind(response)
            while True:
                if cancel is not None and cancel.is_set():
                    raise TurnCancelled()
                line = response.readline()
                if line == b"":
                    break
                stripped = line.strip()
                if stripped == b"":
                    continue
                payload = json.loads(stripped.decode("utf-8"))
                if not isinstance(payload, dict):
                    continue
                if isinstance(payload.get("error"), str):
                    raise ChatModelFailed("The chat model did not answer.")
                content = payload.get("message", {}).get("content", "")
                if isinstance(content, str) and content != "":
                    if buffer != "" and content.startswith(buffer):
                        buffer = content
                    else:
                        buffer += content
                    visible = partial_answer(buffer)
                    if ready_to_show is not None and not ready_to_show(buffer):
                        visible = ""
                    if len(visible) > sent:
                        on_answer_delta(visible[sent:])
                        sent = len(visible)
                if payload.get("done") is True:
                    break
        parsed = json.loads(buffer) if buffer != "" else None
    except TurnCancelled:
        raise
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        if cancel is not None and cancel.is_set():
            raise TurnCancelled() from error
        raise ChatModelFailed("The chat model did not answer.") from error
    if not isinstance(parsed, dict):
        raise ChatModelFailed("The chat model returned an unreadable answer.")
    return parsed
