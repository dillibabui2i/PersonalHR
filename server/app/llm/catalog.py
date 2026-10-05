import json
import re
import threading
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.core.database import open_app_database
from app.core.settings import load_settings
from app.llm.model_status import fetch_installed_model_names, model_is_installed

MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,120}$")


@dataclass(frozen=True)
class CatalogEntry:
    name: str
    kind: str
    description: str
    size: str
    dimensions: int = 0


CATALOGUE: tuple[CatalogEntry, ...] = (
    CatalogEntry("qwen3.5:2b", "chat", "Small chat model already used for answers and planning.", "about 2 GB"),
    CatalogEntry("qwen3:4b", "chat", "Larger Qwen chat model for longer policy answers.", "about 3 GB"),
    CatalogEntry("llama3.2:3b", "chat", "Llama chat model that follows the same answer format.", "about 2 GB"),
    CatalogEntry("gemma3:4b", "chat", "Gemma chat model for the same grounded answers.", "about 3 GB"),
    CatalogEntry("phi4-mini", "chat", "Compact Phi chat model.", "about 2.5 GB"),
    CatalogEntry("mistral:7b", "chat", "Mistral chat model. Needs more memory than the smaller models.", "about 4 GB"),
    CatalogEntry("nomic-embed-text", "embed", "General document embedding model.", "about 270 MB", 768),
    CatalogEntry("mxbai-embed-large", "embed", "Larger embedding model with a longer vector.", "about 670 MB", 1024),
    CatalogEntry("all-minilm", "embed", "Small embedding model with a shorter vector.", "about 50 MB", 384),
    CatalogEntry("embeddinggemma", "embed", "Gemma embedding model.", "about 620 MB", 768),
)

_pulls: dict[str, dict[str, str]] = {}
_pull_lock = threading.Lock()


def catalogue_entry(name: str) -> CatalogEntry | None:
    cleaned = name.strip()
    for entry in CATALOGUE:
        if entry.name == cleaned or cleaned.startswith(f"{entry.name}:"):
            return entry
    return None


def recorded_kind(name: str) -> tuple[str, int] | None:
    with open_app_database() as connection:
        row = connection.execute(
            "SELECT kind, dimensions FROM model_kinds WHERE name = ?",
            (name.strip(),),
        ).fetchone()
    if row is None:
        return None
    return str(row["kind"]), int(row["dimensions"])


def remember_kind(name: str, kind: str, dimensions: int) -> None:
    with open_app_database() as connection:
        connection.execute(
            """
            INSERT INTO model_kinds (name, kind, dimensions)
            VALUES (?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET kind = excluded.kind, dimensions = excluded.dimensions
            """,
            (name.strip(), kind, dimensions),
        )


def model_kind(name: str) -> str:
    recorded = recorded_kind(name)
    if recorded is not None:
        return recorded[0]
    entry = catalogue_entry(name)
    if entry is not None:
        return entry.kind
    if "embed" in name.lower():
        return "embed"
    return "chat"


def installed_names() -> list[str]:
    settings = load_settings()
    try:
        return fetch_installed_model_names(settings.ollama_base_url)
    except (OSError, ValueError, json.JSONDecodeError):
        return []


def ollama_reachable() -> bool:
    settings = load_settings()
    try:
        fetch_installed_model_names(settings.ollama_base_url)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return True


def installed_by_kind(kind: str) -> list[str]:
    return sorted(name for name in installed_names() if model_kind(name) == kind)


def known_dimensions(name: str) -> int:
    recorded = recorded_kind(name)
    if recorded is not None and recorded[1] > 0:
        return recorded[1]
    entry = catalogue_entry(name)
    if entry is not None and entry.dimensions > 0:
        return entry.dimensions
    return 0


def pull_snapshot() -> dict[str, dict[str, str]]:
    with _pull_lock:
        return {name: dict(state) for name, state in _pulls.items()}


def _set_pull(name: str, **fields: str) -> None:
    with _pull_lock:
        state = _pulls.setdefault(name, {"name": name, "kind": "", "status": "pulling", "progress": "", "error": ""})
        state.update(fields)


def start_pull(name: str, kind: str) -> None:
    cleaned = name.strip()
    if MODEL_NAME.fullmatch(cleaned) is None:
        raise ValueError("Enter an Ollama model name, such as llama3.2:3b.")
    if kind not in {"chat", "embed"}:
        raise ValueError("Choose chat or embed.")
    entry = catalogue_entry(cleaned)
    if entry is not None and entry.kind != kind:
        raise ValueError(f"{cleaned} is a {entry.kind} model.")
    with _pull_lock:
        current = _pulls.get(cleaned)
        if current is not None and current.get("status") == "pulling":
            return
    _set_pull(cleaned, kind=kind, status="pulling", progress="Starting download", error="")
    threading.Thread(target=_pull, args=(cleaned, kind), daemon=True).start()


def _pull(name: str, kind: str) -> None:
    settings = load_settings()
    request = Request(
        f"{settings.ollama_base_url.rstrip('/')}/api/pull",
        data=json.dumps({"name": name, "stream": True}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=3600) as response:
            while True:
                line = response.readline()
                if line == b"":
                    break
                payload = json.loads(line.decode("utf-8"))
                if not isinstance(payload, dict):
                    continue
                if isinstance(payload.get("error"), str):
                    raise RuntimeError(payload["error"])
                status = str(payload.get("status", ""))
                total = payload.get("total")
                completed = payload.get("completed")
                if isinstance(total, int) and isinstance(completed, int) and total > 0:
                    status = f"{status} {round(completed * 100 / total)}%"
                if status != "":
                    _set_pull(name, progress=status)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as error:
        detail = "Ollama is not running." if isinstance(error, URLError) else str(error) or "The download failed."
        _set_pull(name, status="failed", error=detail)
        return
    if not model_is_installed(installed_names(), name):
        _set_pull(name, status="failed", error="Ollama finished without installing the model.")
        return
    dimensions = 0
    if kind == "embed":
        try:
            from app.llm.embeddings import EmbeddingFailed, probe_dimensions

            dimensions = probe_dimensions(name)
        except EmbeddingFailed as error:
            _set_pull(name, status="failed", error=error.detail)
            return
    remember_kind(name, kind, dimensions)
    _set_pull(name, status="ready", progress="Installed", error="")


def catalogue_view() -> list[dict[str, object]]:
    installed = installed_names()
    pulls = pull_snapshot()
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for entry in CATALOGUE:
        seen.add(entry.name)
        pull = pulls.get(entry.name, {})
        rows.append(
            {
                "name": entry.name,
                "kind": entry.kind,
                "description": entry.description,
                "size": entry.size,
                "dimensions": entry.dimensions,
                "installed": model_is_installed(installed, entry.name),
                "status": str(pull.get("status", "")),
                "progress": str(pull.get("progress", "")),
                "error": str(pull.get("error", "")),
            }
        )
    for name, pull in pulls.items():
        if name in seen or catalogue_entry(name) is not None:
            continue
        seen.add(name)
        rows.append(
            {
                "name": name,
                "kind": pull.get("kind", "chat"),
                "description": "Model name entered for this workspace.",
                "size": "",
                "dimensions": known_dimensions(name),
                "installed": model_is_installed(installed, name),
                "status": pull.get("status", ""),
                "progress": pull.get("progress", ""),
                "error": pull.get("error", ""),
            }
        )
    for name in installed:
        if any(name == item or name.startswith(f"{item}:") or item.startswith(f"{name}:") for item in seen):
            continue
        rows.append(
            {
                "name": name,
                "kind": model_kind(name),
                "description": "Already installed in Ollama.",
                "size": "",
                "dimensions": known_dimensions(name),
                "installed": True,
                "status": "",
                "progress": "",
                "error": "",
            }
        )
    return rows
