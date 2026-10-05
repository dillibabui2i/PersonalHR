import re

from app.core.settings import load_settings
from app.rag.retrieval.filters import matches_filter, within_distance
from app.rag.types import RetrievedChunk, SearchFilter
from app.rag.vector_store import keyword_chunks, nearest_chunks

MAX_KEYWORD_TERMS = 16
WORD_PATTERN = re.compile(r"[a-z0-9]+")
STOP_WORDS = {
    "a", "an", "and", "are", "also", "can", "do", "does", "for", "from", "get", "have",
    "how", "i", "in", "is", "it", "me", "my", "need", "now", "of", "on", "or", "please",
    "some", "the", "to", "want", "well", "what", "when", "where", "which", "who", "why",
    "will", "with", "you", "your",
}


def compact_query(text: str) -> str:
    words: list[str] = []
    for word in WORD_PATTERN.findall(text.lower()):
        if word in STOP_WORDS or word in words or len(word) < 3:
            continue
        words.append(word)
    return " ".join(words[:MAX_KEYWORD_TERMS])


def keyword_query(text: str) -> str:
    """Prefix terms so a stem such as encash still matches encashment in FTS5."""
    compact = compact_query(text)
    if compact == "":
        return ""
    return " OR ".join(f"{word}*" for word in compact.split())


def passage_key(chunk: RetrievedChunk) -> str:
    return f"{chunk.document_id}\n{' '.join(chunk.text.split())}"


def reciprocal_rank_fusion(ranked_lists: list[list[RetrievedChunk]]) -> list[RetrievedChunk]:
    fusion_constant = load_settings().rag_fusion_constant
    scores: dict[str, float] = {}
    chunks: dict[str, RetrievedChunk] = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            key = passage_key(chunk)
            scores[key] = scores.get(key, 0.0) + 1 / (fusion_constant + rank)
            chunks.setdefault(key, chunk)
    ordered = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [chunks[key] for key in ordered]


def semantic_hits(organization_id: str, vector: list[float], search_filter: SearchFilter) -> list[RetrievedChunk]:
    hits = nearest_chunks(organization_id, vector, load_settings().rag_vector_limit)
    return [hit for hit in hits if matches_filter(hit, search_filter) and within_distance(hit)]


def lexical_hits(organization_id: str, query: str, search_filter: SearchFilter) -> list[RetrievedChunk]:
    hits = keyword_chunks(organization_id, keyword_query(query), load_settings().rag_keyword_limit)
    return [hit for hit in hits if matches_filter(hit, search_filter)]


def hybrid_search(
    organization_id: str,
    queries: list[str],
    vectors: list[list[float]],
    search_filter: SearchFilter,
) -> list[RetrievedChunk]:
    ranked_lists: list[list[RetrievedChunk]] = []
    for query, vector in zip(queries, vectors, strict=True):
        ranked_lists.append(semantic_hits(organization_id, vector, search_filter))
        ranked_lists.append(lexical_hits(organization_id, query, search_filter))
    return reciprocal_rank_fusion(ranked_lists)
