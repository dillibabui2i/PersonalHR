import logging

from app.activity.trace import TurnTrace, open_step
from app.core.settings import load_settings
from app.llm.embeddings import embed_texts
from app.organizations.models import organization_embedding
from app.agent.clarify import prefer_leave_chunks
from app.rag.retrieval.filters import diversify, relevant_by_score
from app.rag.retrieval.hybrid_search import compact_query, hybrid_search
from app.rag.retrieval.reranker import rerank
from app.rag.types import RetrievedChunk, SearchFilter

retrieval_log = logging.getLogger("personal_hr.rag")


def retrieve(
    organization_id: str,
    question: str,
    search_filter: SearchFilter,
    trace: TurnTrace | None = None,
) -> list[RetrievedChunk]:
    if question.strip() == "":
        return []
    settings = load_settings()
    queries = [question]
    compact = compact_query(question)
    if compact != "" and compact.casefold() != question.strip().casefold():
        queries.append(compact)
    lowered = question.lower()
    if "when" in lowered and "day" in lowered:
        for extra in ("leave days credited entitlement accrual", "earned leave credit allotment"):
            if extra not in queries:
                queries.append(extra)
    model, dimensions = organization_embedding(organization_id)
    with open_step(trace, "hybrid search", " | ".join(queries)) as hybrid:
        vectors = embed_texts(queries, "query", model, dimensions)
        candidates = hybrid_search(organization_id, queries, vectors, search_filter)[: settings.rag_candidate_pool]
        hybrid.output_summary = f"{len(candidates)} candidates after fusion."
    with open_step(trace, "rerank", question) as ranked_step:
        ranked = relevant_by_score(rerank(question, candidates))
        top_score = 0.0 if ranked == [] else ranked[0].rerank_score or 0.0
        ranked_step.output_summary = f"{len(candidates)} candidates → {len(ranked)} relevant, top {top_score:.2f}"
    results = prefer_leave_chunks(question, diversify(ranked)[: settings.rag_result_limit])
    retrieval_log.info(
        "retrieve: %d queries, %d candidates, %d after rerank filter, scores %s",
        len(queries),
        len(candidates),
        len(ranked),
        [round(chunk.rerank_score or 0.0, 3) for chunk in results],
    )
    return results
