import json
from dataclasses import dataclass
from urllib.request import Request, urlopen

from app.core.settings import load_settings


@dataclass(frozen=True)
class ModelStatus:
    chat_model: str
    embed_model: str
    chat_ready: bool
    embed_ready: bool
    ollama_reachable: bool
    detail: str


def model_is_installed(installed_names: list[str], expected_name: str) -> bool:
    for installed_name in installed_names:
        if installed_name == expected_name or installed_name.startswith(f"{expected_name}:"):
            return True
    return False


def fetch_installed_model_names(base_url: str) -> list[str]:
    tags_url = f"{base_url.rstrip('/')}/api/tags"
    with urlopen(tags_url, timeout=2) as response:
        payload = json.load(response)
    models = payload.get("models", [])
    if not isinstance(models, list):
        raise ValueError("Ollama did not return a model list.")
    names: list[str] = []
    for model in models:
        if not isinstance(model, dict):
            continue
        name = model.get("name", "")
        if isinstance(name, str) and name != "":
            names.append(name)
    return names


def describe_model_status(
    chat_model: str,
    embed_model: str,
    chat_ready: bool,
    embed_ready: bool,
    ollama_reachable: bool,
) -> str:
    if not ollama_reachable:
        return (
            "Ollama is not running. "
            f"Chat model {chat_model} and embedding model {embed_model} cannot be checked."
        )
    chat_state = "ready" if chat_ready else "missing"
    embed_state = "ready" if embed_ready else "missing"
    return f"Chat model {chat_model} is {chat_state}. Embedding model {embed_model} is {embed_state}."


def read_model_status() -> ModelStatus:
    settings = load_settings()
    chat_model = settings.chat_model
    embed_model = settings.embed_model
    try:
        installed_names = fetch_installed_model_names(settings.ollama_base_url)
    except (OSError, ValueError, json.JSONDecodeError):
        return ModelStatus(
            chat_model=chat_model,
            embed_model=embed_model,
            chat_ready=False,
            embed_ready=False,
            ollama_reachable=False,
            detail=describe_model_status(chat_model, embed_model, False, False, False),
        )

    chat_ready = model_is_installed(installed_names, chat_model)
    embed_ready = model_is_installed(installed_names, embed_model)
    return ModelStatus(
        chat_model=chat_model,
        embed_model=embed_model,
        chat_ready=chat_ready,
        embed_ready=embed_ready,
        ollama_reachable=True,
        detail=describe_model_status(chat_model, embed_model, chat_ready, embed_ready, True),
    )


def post_ollama(base_url: str, path: str, payload: dict[str, object]) -> None:
    request = Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=120):
        pass


def warm_models() -> None:
    """Load configured chat/planner and embedding models and keep them resident."""
    settings = load_settings()
    for model in dict.fromkeys((settings.chat_model, settings.planner_model)):
        post_ollama(
            settings.ollama_base_url,
            "/api/generate",
            {
                "model": model,
                "prompt": "",
                "stream": False,
                "keep_alive": settings.ollama_keep_alive,
                "options": {"num_ctx": settings.ollama_num_ctx},
            },
        )
    post_ollama(
        settings.ollama_base_url,
        "/api/embed",
        {
            "model": settings.embed_model,
            "input": ["search_query: warm up"],
            "keep_alive": settings.ollama_keep_alive,
        },
    )
