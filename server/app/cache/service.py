import hashlib
import json
import re
import uuid
from datetime import datetime, timedelta, timezone

from app.core.database import open_app_database
from app.core.settings import load_settings
from app.rag.types import Citation, RagAnswer


def ready_document_set_key(document_ids: frozenset[str]) -> str:
    joined = "\n".join(sorted(document_ids))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def normalize_question(question: str) -> str:
    normalized = " ".join(question.lower().split())
    return re.sub(r"[.!?]+$", "", normalized).strip()


def citation_payload(citation: Citation) -> dict[str, str]:
    payload = {
        "documentName": citation.document_name,
        "sectionTitle": citation.section_title,
    }
    if citation.excerpt != "":
        payload["excerpt"] = citation.excerpt
    return payload


def parse_citations(raw: str) -> list[Citation]:
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(loaded, list):
        return []
    citations: list[Citation] = []
    for item in loaded:
        if not isinstance(item, dict):
            continue
        document_name = item.get("documentName")
        section_title = item.get("sectionTitle")
        if isinstance(document_name, str) and isinstance(section_title, str):
            excerpt = item.get("excerpt", "")
            citations.append(
                Citation(
                    document_name=document_name,
                    section_title=section_title,
                    excerpt=excerpt if isinstance(excerpt, str) else "",
                )
            )
    return citations


def cache_question_key(question: str, answer_style: str = "brief") -> str:
    base = normalize_question(question)
    if answer_style == "detailed":
        return f"{base}\nstyle:detailed"
    return base


def read_cached_answer(
    organization_id: str,
    document_ids: frozenset[str],
    standalone_question: str,
    answer_style: str = "brief",
) -> RagAnswer | None:
    settings = load_settings()
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=max(0, settings.answer_cache_minutes))
    document_key = ready_document_set_key(document_ids)
    question_key = cache_question_key(standalone_question, answer_style)
    if question_key == "":
        return None
    with open_app_database() as connection:
        connection.execute(
            "DELETE FROM answer_cache WHERE created_at < ?",
            (cutoff.isoformat(),),
        )
        row = connection.execute(
            """
            SELECT answer, citations_json
            FROM answer_cache
            WHERE organization_id = ? AND document_set_key = ? AND question_key = ?
            """,
            (organization_id, document_key, question_key),
        ).fetchone()
    if row is None:
        return None
    return RagAnswer(
        answer=str(row["answer"]),
        citations=parse_citations(str(row["citations_json"])),
        guardrail="Guardrail passed",
    )


def store_cached_answer(
    organization_id: str,
    document_ids: frozenset[str],
    standalone_question: str,
    result: RagAnswer,
    answer_style: str = "brief",
) -> None:
    question_key = cache_question_key(standalone_question, answer_style)
    if question_key == "":
        return
    citations_json = json.dumps([citation_payload(citation) for citation in result.citations])
    with open_app_database() as connection:
        connection.execute(
            """
            INSERT INTO answer_cache (
                id, organization_id, document_set_key, question_key, answer, citations_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (organization_id, document_set_key, question_key)
            DO UPDATE SET answer = excluded.answer,
                          citations_json = excluded.citations_json,
                          created_at = excluded.created_at
            """,
            (
                str(uuid.uuid4()),
                organization_id,
                ready_document_set_key(document_ids),
                question_key,
                result.answer,
                citations_json,
                datetime.now(timezone.utc).isoformat(),
            ),
        )


def invalidate_organization_cache(organization_id: str) -> None:
    with open_app_database() as connection:
        connection.execute("DELETE FROM answer_cache WHERE organization_id = ?", (organization_id,))
