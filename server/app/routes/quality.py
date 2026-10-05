import asyncio

from fastapi import APIRouter, BackgroundTasks, Header, Response
from pydantic import BaseModel, ConfigDict, Field

from app.admin.session import require_organization_access
from app.quality.service import (
    QualityCheck,
    QualityResult,
    QualityRunState,
    begin_quality_run,
    create_quality_check,
    delete_quality_check,
    execute_quality_run,
    list_quality_checks,
    quality_run_state,
    run_quality_check,
    update_quality_check,
)
from app.routes.chat import CitationResponse, citation_response

router = APIRouter()


class QualityCheckBody(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    question: str = ""
    expected_phrase: str = Field(default="", validation_alias="expectedPhrase")
    expected_document: str = Field(default="", validation_alias="expectedDocument")
    expect_no_answer: bool = Field(default=False, validation_alias="expectNoAnswer")


class QualityResultResponse(BaseModel):
    passed: bool
    answer: str
    citations: list[CitationResponse]
    total_ms: int = Field(serialization_alias="totalMs")
    engine: str
    created_at: str = Field(serialization_alias="createdAt")
    guardrail: str


class QualityRunResponse(BaseModel):
    status: str
    detail: str
    current_check_id: str = Field(serialization_alias="currentCheckId")
    waiting_check_ids: list[str] = Field(serialization_alias="waitingCheckIds")


class QualityCheckResponse(BaseModel):
    id: str
    question: str
    expected_phrase: str = Field(serialization_alias="expectedPhrase")
    expected_document: str = Field(serialization_alias="expectedDocument")
    expect_no_answer: bool = Field(serialization_alias="expectNoAnswer")
    latest_result: QualityResultResponse | None = Field(serialization_alias="latestResult")


class QualityCheckListResponse(BaseModel):
    status: str
    detail: str
    current_check_id: str = Field(serialization_alias="currentCheckId")
    waiting_check_ids: list[str] = Field(serialization_alias="waitingCheckIds")
    checks: list[QualityCheckResponse]


def result_response(result: QualityResult) -> QualityResultResponse:
    return QualityResultResponse(
        passed=result.passed,
        answer=result.answer,
        citations=[citation_response(citation) for citation in result.citations],
        total_ms=result.total_ms,
        engine=result.engine,
        created_at=result.created_at,
        guardrail=result.guardrail,
    )


def run_response(state: QualityRunState) -> QualityRunResponse:
    return QualityRunResponse(
        status=state.status,
        detail=state.detail,
        current_check_id=state.current_check_id,
        waiting_check_ids=state.waiting_check_ids,
    )


def check_response(check: QualityCheck) -> QualityCheckResponse:
    latest = None if check.latest_result is None else result_response(check.latest_result)
    return QualityCheckResponse(
        id=check.id,
        question=check.question,
        expected_phrase=check.expected_phrase,
        expected_document=check.expected_document,
        expect_no_answer=check.expect_no_answer,
        latest_result=latest,
    )


@router.get("/admin/organizations/{organization_id}/quality-checks", response_model=QualityCheckListResponse)
def read_quality_checks(
    organization_id: str,
    x_admin_session: str = Header(default=""),
) -> QualityCheckListResponse:
    require_organization_access(x_admin_session, organization_id)
    state = quality_run_state(organization_id)
    return QualityCheckListResponse(
        status=state.status,
        detail=state.detail,
        current_check_id=state.current_check_id,
        waiting_check_ids=state.waiting_check_ids,
        checks=[check_response(check) for check in list_quality_checks(organization_id)],
    )


async def run_checks_after_response(run_id: str) -> None:
    await asyncio.to_thread(execute_quality_run, run_id)


@router.post("/admin/organizations/{organization_id}/quality-checks/run", response_model=QualityRunResponse)
def start_quality_checks(
    organization_id: str,
    background_tasks: BackgroundTasks,
    x_admin_session: str = Header(default=""),
) -> QualityRunResponse:
    require_organization_access(x_admin_session, organization_id)
    run_id = begin_quality_run(organization_id)
    background_tasks.add_task(run_checks_after_response, run_id)
    return run_response(quality_run_state(organization_id))


@router.post(
    "/admin/organizations/{organization_id}/quality-checks",
    response_model=QualityCheckResponse,
    status_code=201,
)
def add_quality_check(
    organization_id: str,
    body: QualityCheckBody,
    x_admin_session: str = Header(default=""),
) -> QualityCheckResponse:
    require_organization_access(x_admin_session, organization_id)
    check = create_quality_check(
        organization_id,
        body.question,
        body.expected_phrase,
        body.expected_document,
        body.expect_no_answer,
    )
    return check_response(check)


@router.patch(
    "/admin/organizations/{organization_id}/quality-checks/{check_id}",
    response_model=QualityCheckResponse,
)
def edit_quality_check(
    organization_id: str,
    check_id: str,
    body: QualityCheckBody,
    x_admin_session: str = Header(default=""),
) -> QualityCheckResponse:
    require_organization_access(x_admin_session, organization_id)
    check = update_quality_check(
        organization_id,
        check_id,
        body.question,
        body.expected_phrase,
        body.expected_document,
        body.expect_no_answer,
    )
    return check_response(check)


@router.delete("/admin/organizations/{organization_id}/quality-checks/{check_id}", status_code=204)
def remove_quality_check(
    organization_id: str,
    check_id: str,
    x_admin_session: str = Header(default=""),
) -> Response:
    require_organization_access(x_admin_session, organization_id)
    delete_quality_check(organization_id, check_id)
    return Response(status_code=204)


@router.post(
    "/admin/organizations/{organization_id}/quality-checks/{check_id}/run",
    response_model=QualityCheckResponse,
)
def run_one_quality_check(
    organization_id: str,
    check_id: str,
    x_admin_session: str = Header(default=""),
) -> QualityCheckResponse:
    require_organization_access(x_admin_session, organization_id)
    return check_response(run_quality_check(organization_id, check_id))
