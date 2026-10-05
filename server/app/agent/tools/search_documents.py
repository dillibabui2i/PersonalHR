from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from app.activity.trace import open_step, raise_if_cancelled
from app.agent.context import active_turn
from app.llm.embeddings import EmbeddingFailed
from app.rag.glossary import spell_out_terms
from app.rag.retrieval.retriever import retrieve
from app.rag.types import SearchFilter
from app.rag.vector_store import read_terms


class SearchInput(BaseModel):
    question: str = Field(description="The employee's question to search in the uploaded documents.")


class SearchDocumentsTool(BaseTool):
    name: str = "search_documents"
    description: str = (
        "Search this organization's uploaded documents. "
        "Use it for any question about rules, eligibility, or policy text."
    )
    args_schema: Type[BaseModel] = SearchInput

    def _run(self, question: str) -> str:
        return self.execute(question)

    def execute(self, question: str) -> str:
        context = active_turn.get()
        if context is None:
            return "Search is not available."
        if context.searched:
            return context.search_text
        raise_if_cancelled(context.trace)
        context.searched = True
        spelled = spell_out_terms(question.strip(), read_terms(context.organization_id))
        with open_step(context.trace, "search_documents", spelled, "tool") as outcome:
            try:
                chunks = retrieve(
                    context.organization_id,
                    spelled,
                    SearchFilter(document_ids=context.document_ids),
                    context.trace,
                )
            except EmbeddingFailed:
                context.search_failed = True
                outcome.status = "failed"
                outcome.output_summary = "Search could not run."
                context.search_text = "Search could not run."
                return context.search_text
            with context.result_lock:
                if not context.accept_results:
                    outcome.output_summary = "Search is unavailable."
                    return "Search is unavailable."
                context.chunks = chunks
                outcome.output_summary = f"{len(chunks)} excerpts"
                if chunks == []:
                    context.search_text = "No excerpts matched."
                    return context.search_text
                lines = [
                    f"[{index}] {chunk.document_name} — {chunk.section_title}\n{chunk.text}"
                    for index, chunk in enumerate(chunks, start=1)
                ]
                context.search_text = "\n\n".join(lines)
                return context.search_text
