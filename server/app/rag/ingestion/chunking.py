import math
import re

from app.core.settings import Settings, load_settings
from app.llm.embeddings import embed_texts
from app.rag.ingestion.sections import Section, split_sections
from app.rag.types import Chunk

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\s*[\u2022\uf0b7\u25aa\u25cf]\s*")


def split_sentences(text: str) -> list[str]:
    flattened = re.sub(r"[ \t]*\n[ \t]*", " ", text)
    return [part.strip() for part in SENTENCE_BOUNDARY.split(flattened) if part.strip() != ""]


def cosine_distance(first: list[float], second: list[float]) -> float:
    dot = sum(a * b for a, b in zip(first, second, strict=True))
    norm = math.sqrt(sum(a * a for a in first)) * math.sqrt(sum(b * b for b in second))
    if norm == 0:
        return 1.0
    return 1.0 - dot / norm


def percentile(values: list[float], amount: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * amount / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def group_sentences(sentences: list[str], distances: list[float], settings: Settings) -> list[str]:
    threshold = percentile(distances, settings.chunk_breakpoint_percentile) if distances != [] else 1.0
    groups: list[str] = []
    current: list[str] = [sentences[0]]
    for index, sentence in enumerate(sentences[1:]):
        current_text = " ".join(current)
        topic_shift = distances[index] > threshold and len(current_text) >= settings.chunk_min_characters
        too_long = len(current_text) + len(sentence) + 1 > settings.chunk_max_characters
        if topic_shift or too_long:
            groups.append(current_text)
            current = [sentence]
            continue
        current.append(sentence)
    groups.append(" ".join(current))
    return groups


def chunk_section(section: Section, model: str, dimensions: int) -> list[Chunk]:
    sentences = split_sentences(section.body)
    if sentences == []:
        return []
    settings = load_settings()
    joined = " ".join(sentences)
    if len(joined) <= settings.chunk_max_characters or len(sentences) == 1:
        return [Chunk(section_title=section.title, text=joined)]
    vectors = embed_texts(
        [f"{section.title}\n{sentence}" for sentence in sentences],
        "document",
        model,
        dimensions,
    )
    distances = [cosine_distance(vectors[index], vectors[index + 1]) for index in range(len(vectors) - 1)]
    groups = group_sentences(sentences, distances, settings)
    return [Chunk(section_title=section.title, text=text) for text in groups]


def build_chunks(text: str, fallback_title: str, model: str, dimensions: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in split_sections(text, fallback_title):
        chunks.extend(chunk_section(section, model, dimensions))
    return chunks
