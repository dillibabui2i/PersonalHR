import json
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from fastapi import HTTPException

quality_log = logging.getLogger("personal_hr.quality")

from app.core.database import open_app_database
from app.documents.service import ready_document_ids
from app.organizations.service import read_organization
from app.agent.engine import answer_with_engine, assistant_engine_name
from app.rag.prompts import NO_ANSWER
from app.rag.types import Citation, SearchFilter

STARTER_HOST = "iassistant.ideas2it.com"


@dataclass(frozen=True)
class QualityResult:
    passed: bool
    answer: str
    citations: list[Citation]
    total_ms: int
    engine: str
    created_at: str
    guardrail: str


@dataclass(frozen=True)
class QualityCheck:
    id: str
    question: str
    expected_phrase: str
    expected_document: str
    expect_no_answer: bool
    latest_result: QualityResult | None


@dataclass(frozen=True)
class CheckDraft:
    question: str
    expected_phrase: str
    expected_document: str
    expect_no_answer: bool


def clean_check_draft(question: str, expected_phrase: str, expected_document: str, expect_no_answer: bool) -> CheckDraft:
    cleaned_question = question.strip()
    if cleaned_question == "":
        raise HTTPException(status_code=400, detail="Enter a question.")
    if expect_no_answer:
        return CheckDraft(cleaned_question, "", "", True)
    cleaned_phrase = expected_phrase.strip()
    if cleaned_phrase == "":
        raise HTTPException(status_code=400, detail="Enter the phrase you expect, or mark the check as no answer.")
    return CheckDraft(cleaned_question, cleaned_phrase, expected_document.strip(), False)


def passes_check(check: CheckDraft, answer: str, citations: list[Citation]) -> bool:
    if check.expect_no_answer:
        return answer == NO_ANSWER
    if check.expected_phrase.lower() not in answer.lower():
        return False
    document_name = check.expected_document.lower()
    if document_name == "":
        return True
    return any(citation.document_name.lower() == document_name for citation in citations)


def citations_from_json(raw: str) -> list[Citation]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    citations: list[Citation] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        document_name = item.get("documentName", "")
        section_title = item.get("sectionTitle", "")
        if isinstance(document_name, str) and isinstance(section_title, str):
            citations.append(Citation(document_name=document_name, section_title=section_title))
    return citations


def result_from_row(row: object) -> QualityResult:
    record = row
    return QualityResult(
        passed=bool(record["passed"]),
        answer=str(record["answer"]),
        citations=citations_from_json(str(record["citations_json"])),
        total_ms=int(record["total_ms"]),
        engine=str(record["engine"]),
        created_at=str(record["created_at"]),
        guardrail=str(record["guardrail"]),
    )


def latest_result(check_id: str) -> QualityResult | None:
    with open_app_database() as connection:
        row = connection.execute(
            """
            SELECT passed, answer, citations_json, total_ms, engine, created_at, guardrail
            FROM quality_results
            WHERE check_id = ?
            ORDER BY created_at DESC, rowid DESC
            LIMIT 1
            """,
            (check_id,),
        ).fetchone()
    if row is None:
        return None
    return result_from_row(row)


def starter_drafts() -> list[CheckDraft]:
    return []


def ensure_starter_checks(organization_id: str) -> None:
    organization = read_organization(organization_id)
    if organization.site_host != STARTER_HOST:
        return
    with open_app_database() as connection:
        row = connection.execute(
            "SELECT COUNT(*) AS count FROM quality_checks WHERE organization_id = ?",
            (organization.id,),
        ).fetchone()
        if int(row["count"]) > 0:
            return
        created_at = datetime.now(timezone.utc).isoformat()
        for draft in starter_drafts():
            connection.execute(
                """
                INSERT INTO quality_checks (
                    id, organization_id, question, expected_phrase, expected_document, expect_no_answer, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    organization.id,
                    draft.question,
                    draft.expected_phrase,
                    draft.expected_document,
                    1 if draft.expect_no_answer else 0,
                    created_at,
                ),
            )


def list_quality_checks(organization_id: str) -> list[QualityCheck]:
    ensure_starter_checks(organization_id)
    with open_app_database() as connection:
        rows = connection.execute(
            """
            SELECT id, question, expected_phrase, expected_document, expect_no_answer
            FROM quality_checks
            WHERE organization_id = ?
            ORDER BY created_at, rowid
            """,
            (organization_id,),
        ).fetchall()
    return [
        QualityCheck(
            id=str(row["id"]),
            question=str(row["question"]),
            expected_phrase=str(row["expected_phrase"]),
            expected_document=str(row["expected_document"]),
            expect_no_answer=bool(row["expect_no_answer"]),
            latest_result=latest_result(str(row["id"])),
        )
        for row in rows
    ]


def insert_check(organization_id: str, draft: CheckDraft) -> str:
    check_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    with open_app_database() as connection:
        connection.execute(
            """
            INSERT INTO quality_checks (
                id, organization_id, question, expected_phrase, expected_document, expect_no_answer, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                check_id,
                organization_id,
                draft.question,
                draft.expected_phrase,
                draft.expected_document,
                1 if draft.expect_no_answer else 0,
                created_at,
            ),
        )
    return check_id


def create_quality_check(
    organization_id: str,
    question: str,
    expected_phrase: str,
    expected_document: str,
    expect_no_answer: bool,
) -> QualityCheck:
    read_organization(organization_id)
    draft = clean_check_draft(question, expected_phrase, expected_document, expect_no_answer)
    check_id = insert_check(organization_id, draft)
    return read_quality_check(organization_id, check_id)


def update_quality_check(
    organization_id: str,
    check_id: str,
    question: str,
    expected_phrase: str,
    expected_document: str,
    expect_no_answer: bool,
) -> QualityCheck:
    read_quality_check(organization_id, check_id)
    draft = clean_check_draft(question, expected_phrase, expected_document, expect_no_answer)
    with open_app_database() as connection:
        connection.execute(
            """
            UPDATE quality_checks
            SET question = ?, expected_phrase = ?, expected_document = ?, expect_no_answer = ?
            WHERE id = ? AND organization_id = ?
            """,
            (
                draft.question,
                draft.expected_phrase,
                draft.expected_document,
                1 if draft.expect_no_answer else 0,
                check_id,
                organization_id,
            ),
        )
    return read_quality_check(organization_id, check_id)


def delete_quality_check(organization_id: str, check_id: str) -> None:
    read_quality_check(organization_id, check_id)
    with open_app_database() as connection:
        connection.execute("DELETE FROM quality_results WHERE check_id = ?", (check_id,))
        connection.execute(
            "DELETE FROM quality_checks WHERE id = ? AND organization_id = ?",
            (check_id, organization_id),
        )


def read_quality_check(organization_id: str, check_id: str) -> QualityCheck:
    with open_app_database() as connection:
        row = connection.execute(
            """
            SELECT id, question, expected_phrase, expected_document, expect_no_answer
            FROM quality_checks
            WHERE id = ? AND organization_id = ?
            """,
            (check_id, organization_id),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="That quality check was not found.")
    return QualityCheck(
        id=str(row["id"]),
        question=str(row["question"]),
        expected_phrase=str(row["expected_phrase"]),
        expected_document=str(row["expected_document"]),
        expect_no_answer=bool(row["expect_no_answer"]),
        latest_result=latest_result(check_id),
    )


def store_result(
    check_id: str,
    passed: bool,
    answer: str,
    citations: list[Citation],
    total_ms: int,
    guardrail: str,
) -> None:
    payload = [
        {"documentName": citation.document_name, "sectionTitle": citation.section_title}
        for citation in citations
    ]
    with open_app_database() as connection:
        connection.execute(
            """
            INSERT INTO quality_results (
                id, check_id, engine, passed, answer, citations_json, total_ms, created_at, guardrail
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                check_id,
                assistant_engine_name(),
                1 if passed else 0,
                answer,
                json.dumps(payload),
                total_ms,
                datetime.now(timezone.utc).isoformat(),
                guardrail,
            ),
        )


@dataclass(frozen=True)
class QualityRunState:
    status: str
    detail: str
    current_check_id: str
    waiting_check_ids: list[str]


def quality_run_state(organization_id: str) -> QualityRunState:
    with open_app_database() as connection:
        row = connection.execute(
            """
            SELECT status, detail, check_ids_json, current_index
            FROM quality_runs
            WHERE organization_id = ?
            ORDER BY created_at DESC, rowid DESC
            LIMIT 1
            """,
            (organization_id,),
        ).fetchone()
    if row is None or str(row["status"]) == "done":
        return QualityRunState("idle", "", "", [])
    if str(row["status"]) != "running":
        return QualityRunState("stopped", str(row["detail"]), "", [])
    parsed = json.loads(str(row["check_ids_json"]))
    check_ids = [str(item) for item in parsed] if isinstance(parsed, list) else []
    index = int(row["current_index"])
    if index < 0 or index >= len(check_ids):
        return QualityRunState("running", "", "", [])
    return QualityRunState("running", "", check_ids[index], check_ids[index + 1 :])


def begin_quality_run(organization_id: str) -> str:
    organization = read_organization(organization_id)
    checks = list_quality_checks(organization.id)
    if checks == []:
        raise HTTPException(status_code=400, detail="Add a check before running.")
    run_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    with open_app_database() as connection:
        connection.execute("BEGIN IMMEDIATE")
        existing = connection.execute(
            "SELECT id FROM quality_runs WHERE organization_id = ? AND status = 'running'",
            (organization.id,),
        ).fetchone()
        if existing is not None:
            raise HTTPException(status_code=409, detail="Checks are already running.")
        connection.execute(
            """
            INSERT INTO quality_runs (
                id, organization_id, status, check_ids_json, current_index, detail, created_at, finished_at
            )
            VALUES (?, ?, 'running', ?, 0, '', ?, '')
            """,
            (run_id, organization.id, json.dumps([check.id for check in checks]), created_at),
        )
    return run_id


def finish_quality_run(run_id: str, status: str, detail: str) -> None:
    with open_app_database() as connection:
        connection.execute(
            """
            UPDATE quality_runs
            SET status = ?, detail = ?, finished_at = ?
            WHERE id = ? AND status = 'running'
            """,
            (status, detail, datetime.now(timezone.utc).isoformat(), run_id),
        )


def stop_abandoned_runs() -> None:
    with open_app_database() as connection:
        connection.execute(
            """
            UPDATE quality_runs
            SET status = 'stopped',
                detail = 'The run stopped because the assistant restarted.',
                finished_at = ?
            WHERE status = 'running'
            """,
            (datetime.now(timezone.utc).isoformat(),),
        )


def execute_quality_run(run_id: str) -> None:
    with open_app_database() as connection:
        row = connection.execute(
            """
            SELECT organization_id, check_ids_json
            FROM quality_runs
            WHERE id = ? AND status = 'running'
            """,
            (run_id,),
        ).fetchone()
    if row is None:
        return
    organization_id = str(row["organization_id"])
    parsed = json.loads(str(row["check_ids_json"]))
    check_ids = [str(item) for item in parsed] if isinstance(parsed, list) else []
    try:
        for index, check_id in enumerate(check_ids):
            with open_app_database() as connection:
                connection.execute(
                    """
                    UPDATE quality_runs
                    SET current_index = ?
                    WHERE id = ? AND status = 'running'
                    """,
                    (index, run_id),
                )
            try:
                run_quality_check(organization_id, check_id)
            except HTTPException:
                continue
        finish_quality_run(run_id, "done", "")
    except Exception:
        quality_log.exception("quality run %s stopped", run_id)
        finish_quality_run(run_id, "stopped", "The run stopped before every check finished.")


def run_quality_check(organization_id: str, check_id: str) -> QualityCheck:
    organization = read_organization(organization_id)
    check = read_quality_check(organization.id, check_id)
    draft = CheckDraft(check.question, check.expected_phrase, check.expected_document, check.expect_no_answer)
    started = time.perf_counter()
    result = answer_with_engine(
        organization.id,
        check.question,
        SearchFilter(document_ids=ready_document_ids(organization.id)),
        None,
        "",
        None,
        str(uuid.uuid4()),
    )
    total_ms = int((time.perf_counter() - started) * 1000)
    store_result(
        check.id,
        passes_check(draft, result.answer, result.citations),
        result.answer,
        result.citations,
        total_ms,
        result.guardrail,
    )
    return read_quality_check(organization.id, check.id)
