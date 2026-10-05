import json
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.core.settings import load_settings

EmbeddingPurpose = Literal["document", "query"]
PURPOSE_PREFIXES: dict[EmbeddingPurpose, str] = {
    "document": "search_document: ",
    "query": "search_query: ",
}


class EmbeddingFailed(Exception):
    def __init__(self, detail: str) -> None:
        self.detail = detail


def embedding_input(text: str, purpose: EmbeddingPurpose, model: str) -> str:
    """Nomic models expect a search prefix. Other embedding models take the text as written."""
    if model.split(":", 1)[0] == "nomic-embed-text":
        return f"{PURPOSE_PREFIXES[purpose]}{text}"
    return text


def probe_dimensions(model: str) -> int:
    vectors = embed_texts(["dimension check"], "document", model, dimensions=0)
    return len(vectors[0])


def embed_texts(
    texts: list[str],
    purpose: EmbeddingPurpose,
    model: str | None = None,
    dimensions: int | None = None,
) -> list[list[float]]:
    if texts == []:
        return []
    settings = load_settings()
    chosen = settings.embed_model if model is None or model.strip() == "" else model.strip()
    expected = settings.embed_dimensions if dimensions is None else dimensions
    request = Request(
        f"{settings.ollama_base_url.rstrip('/')}/api/embed",
        data=json.dumps(
            {
                "model": chosen,
                "input": [embedding_input(text, purpose, chosen) for text in texts],
                "keep_alive": settings.ollama_keep_alive,
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            payload = json.load(response)
    except HTTPError as error:
        if error.code == 404:
            raise EmbeddingFailed(f"Embedding model {chosen} is missing.") from error
        raise EmbeddingFailed("The file could not be embedded.") from error
    except (URLError, TimeoutError, json.JSONDecodeError) as error:
        raise EmbeddingFailed(
            f"Ollama is not running, so embedding model {chosen} could not be used."
        ) from error

    vectors = payload.get("embeddings")
    if not isinstance(vectors, list) or len(vectors) != len(texts):
        raise EmbeddingFailed("The file could not be embedded.")
    parsed: list[list[float]] = []
    for vector in vectors:
        if not isinstance(vector, list) or vector == []:
            raise EmbeddingFailed("The file could not be embedded.")
        if expected > 0 and len(vector) != expected:
            raise EmbeddingFailed("The file could not be embedded.")
        parsed.append([float(value) for value in vector])
    return parsed
