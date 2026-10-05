from app.core.settings import load_settings
from app.rag.types import RetrievedChunk, SearchFilter


def matches_filter(chunk: RetrievedChunk, search_filter: SearchFilter) -> bool:
    if search_filter.document_ids is None:
        return True
    return chunk.document_id in search_filter.document_ids


def within_distance(chunk: RetrievedChunk) -> bool:
    if chunk.vector_distance is None:
        return True
    return chunk.vector_distance <= load_settings().rag_max_vector_distance


def relevant_by_score(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    if chunks == []:
        return []
    settings = load_settings()
    best_score = chunks[0].rerank_score or 0.0
    floor = max(settings.rag_min_rerank_score, best_score * settings.rag_min_relative_rerank_score)
    return [chunk for chunk in chunks if (chunk.rerank_score or 0.0) >= floor]


def diversify(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    limit = load_settings().rag_max_chunks_per_section
    per_section: dict[tuple[str, str], int] = {}
    kept: list[RetrievedChunk] = []
    for chunk in chunks:
        key = (chunk.document_id, chunk.section_title)
        if per_section.get(key, 0) >= limit:
            continue
        per_section[key] = per_section.get(key, 0) + 1
        kept.append(chunk)
    return kept
