from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    section_title: str
    text: str


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    document_name: str
    section_title: str
    text: str
    vector_distance: float | None = None
    rerank_score: float | None = None


@dataclass(frozen=True)
class SearchFilter:
    document_ids: frozenset[str] | None = None


EXCERPT_LIMIT = 600


def excerpt_text(text: str) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= EXCERPT_LIMIT:
        return cleaned
    return f"{cleaned[: EXCERPT_LIMIT - 1].rstrip()}…"


@dataclass(frozen=True)
class Citation:
    document_name: str
    section_title: str
    excerpt: str = ""


def citation_for_chunk(chunk: RetrievedChunk) -> Citation:
    return Citation(
        document_name=chunk.document_name,
        section_title=chunk.section_title,
        excerpt=excerpt_text(chunk.text),
    )


@dataclass(frozen=True)
class RagAnswer:
    answer: str
    citations: list[Citation]
    guardrail: str = ""
    choices: tuple[str, ...] = ()
    suggestions: tuple[str, ...] = ()
    from_page: bool = False
