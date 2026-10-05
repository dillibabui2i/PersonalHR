from dataclasses import replace
from functools import lru_cache
from typing import Any

from app.core.settings import load_settings
from app.rag.types import RetrievedChunk


@lru_cache(maxsize=1)
def load_reranker() -> Any:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(load_settings().reranker_model, device="cpu")


def rerank(question: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    if chunks == []:
        return []
    pairs = [[question, f"{chunk.section_title}\n{chunk.text}"] for chunk in chunks]
    scores = load_reranker().predict(pairs, show_progress_bar=False)
    scored = [replace(chunk, rerank_score=float(score)) for chunk, score in zip(chunks, scores, strict=True)]
    return sorted(scored, key=lambda chunk: chunk.rerank_score or 0.0, reverse=True)
