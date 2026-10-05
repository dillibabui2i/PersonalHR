from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from app.chat.turn import take_chat_turn
from app.extension.access import require_extension_key
from app.extension.sites import registration_for_host
from app.extension.snapshots import save_snapshot
from app.organizations.service import normalize_site_host
from app.organizations.service import find_organization_by_host
from app.employees.records import (
    EmployeeRecord,
    delete_employee,
    ensure_employee,
    ensure_record_facts,
    forget_notes,
    normalize_email,
    save_onboarding,
    scoped_client_id,
)
from app.memory.profile import profile_email
from app.memory.service import (
    forget_all_facts,
    forget_fact,
    profile_loaded,
    recall_facts,
)
from app.core.settings import load_settings
from app.agent.streaming import stop_turn
from app.routes.chat import (
    ChatRequest,
    ChatResponse,
    FeedbackRequest,
    FeedbackResponse,
    HistoryResponse,
    ResetResponse,
    chat_response,
    decline_chat,
    history_response,
    rate_answer,
    reset_chat,
    stream_chat,
)

router = APIRouter()


class SiteResponse(BaseModel):
    registered: bool
    organization_name: str | None = Field(default=None, serialization_alias="organizationName")
    portal_enabled: bool | None = Field(default=None, serialization_alias="portalEnabled")
    memory_enabled: bool | None = Field(default=None, serialization_alias="memoryEnabled")
    profile_api_path: str | None = Field(default=None, serialization_alias="profileApiPath")


class MemoryFactResponse(BaseModel):
    id: str
    fact: str
    kind: str
    created_at: str = Field(serialization_alias="createdAt")


class IdentityRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    client_id: str = Field(default="", validation_alias="clientId")
    site_host: str = Field(default="", validation_alias="siteHost")
    employee_email: str = Field(default="", validation_alias="employeeEmail")
    profile: object | None = None


@router.get("/extension/sites", response_model=SiteResponse, response_model_exclude_none=True)
def read_site_registration(
    host: str = Query(default=""),
    x_extension_key: str = Header(default=""),
) -> SiteResponse:
    require_extension_key(x_extension_key)
    registration = registration_for_host(host)
    if not registration.registered:
        return SiteResponse(registered=False)
    return SiteResponse(
        registered=True,
        organization_name=registration.organization_name,
        portal_enabled=registration.portal_enabled,
        memory_enabled=registration.memory_enabled,
        profile_api_path=registration.profile_api_path,
    )


@router.get("/extension/identity")
def extension_identity_status(
    client_id: str = Query(default="", alias="clientId"),
    site_host: str = Query(default="", alias="siteHost"),
    employee_email: str = Query(default="", alias="employeeEmail"),
    x_extension_key: str = Header(default=""),
) -> dict[str, bool]:
    require_extension_key(x_extension_key)
    organization = find_organization_by_host(site_host)
    scope = scoped_client_id(client_id, site_host, employee_email)
    return {"loaded": profile_loaded(organization.id, scope)}


@router.post("/extension/identity")
def extension_save_identity(
    body: IdentityRequest,
    x_extension_key: str = Header(default=""),
) -> dict[str, bool | int | str]:
    require_extension_key(x_extension_key)
    if body.client_id.strip() == "":
        raise HTTPException(status_code=400, detail="A browser id is required.")
    if body.profile is None:
        raise HTTPException(status_code=400, detail="Send the profile API response.")
    email = normalize_email(body.employee_email) or profile_email(body.profile)
    if email == "":
        raise HTTPException(status_code=422, detail="The profile did not include an email.")
    organization = find_organization_by_host(body.site_host)
    ensure_employee(organization.id, email)
    return {"saved": True, "count": 1, "email": email}


def employee_payload(employee: EmployeeRecord) -> dict[str, str | bool]:
    return {
        "email": employee.email,
        "onboarded": employee.onboarded_at != "",
        "displayName": employee.display_name,
        "roleTitle": employee.role_title,
        "joiningDate": employee.joining_date,
        "workLocation": employee.work_location,
    }


class OnboardingRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    site_host: str = Field(default="", validation_alias="siteHost")
    employee_email: str = Field(default="", validation_alias="employeeEmail")
    display_name: str = Field(default="", validation_alias="displayName")
    role_title: str = Field(default="", validation_alias="roleTitle")
    joining_date: str = Field(default="", validation_alias="joiningDate")
    work_location: str = Field(default="", validation_alias="workLocation")


@router.get("/extension/employee")
def extension_employee(
    site_host: str = Query(default="", alias="siteHost"),
    employee_email: str = Query(default="", alias="employeeEmail"),
    x_extension_key: str = Header(default=""),
) -> dict[str, str | bool]:
    require_extension_key(x_extension_key)
    email = normalize_email(employee_email)
    if email == "":
        raise HTTPException(status_code=400, detail="The profile did not include an email.")
    organization = find_organization_by_host(site_host)
    return employee_payload(ensure_employee(organization.id, email))


@router.post("/extension/employee")
def extension_save_onboarding(
    body: OnboardingRequest,
    x_extension_key: str = Header(default=""),
) -> dict[str, str | bool]:
    require_extension_key(x_extension_key)
    email = normalize_email(body.employee_email)
    if email == "":
        raise HTTPException(status_code=400, detail="The profile did not include an email.")
    organization = find_organization_by_host(body.site_host)
    try:
        employee = save_onboarding(
            organization.id,
            email,
            body.display_name,
            body.role_title,
            body.joining_date,
            body.work_location,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return employee_payload(employee)


@router.delete("/extension/employee")
def extension_delete_employee(
    site_host: str = Query(default="", alias="siteHost"),
    employee_email: str = Query(default="", alias="employeeEmail"),
    x_extension_key: str = Header(default=""),
) -> dict[str, bool]:
    require_extension_key(x_extension_key)
    email = normalize_email(employee_email)
    if email == "":
        raise HTTPException(status_code=400, detail="The profile did not include an email.")
    organization = find_organization_by_host(site_host)
    return {"removed": delete_employee(organization.id, email)}


@router.get("/extension/memory", response_model=list[MemoryFactResponse])
def extension_memory(
    client_id: str = Query(default="", alias="clientId"),
    site_host: str = Query(default="", alias="siteHost"),
    employee_email: str = Query(default="", alias="employeeEmail"),
    x_extension_key: str = Header(default=""),
) -> list[MemoryFactResponse]:
    require_extension_key(x_extension_key)
    organization = find_organization_by_host(site_host)
    scope = scoped_client_id(client_id, site_host, employee_email)
    employee = None if normalize_email(employee_email) == "" else ensure_employee(organization.id, employee_email)
    if employee is not None and employee.onboarded_at != "":
        ensure_record_facts(employee)
    return [
        MemoryFactResponse(id=fact.id, fact=fact.fact, kind=fact.kind, created_at=fact.created_at)
        for fact in recall_facts(
            organization.id, scope, load_settings().employee_fact_limit, conversation_only=True
        )
    ]


@router.delete("/extension/memory")
def extension_forget_all(
    client_id: str = Query(default="", alias="clientId"),
    site_host: str = Query(default="", alias="siteHost"),
    employee_email: str = Query(default="", alias="employeeEmail"),
    x_extension_key: str = Header(default=""),
) -> dict[str, int]:
    require_extension_key(x_extension_key)
    organization = find_organization_by_host(site_host)
    scope = scoped_client_id(client_id, site_host, employee_email)
    if normalize_email(employee_email) == "":
        return {"forgotten": forget_all_facts(organization.id, scope)}
    return {"forgotten": forget_notes(organization.id, scope)}


@router.delete("/extension/memory/{fact_id}")
def extension_forget_fact(
    fact_id: str,
    client_id: str = Query(default="", alias="clientId"),
    site_host: str = Query(default="", alias="siteHost"),
    employee_email: str = Query(default="", alias="employeeEmail"),
    x_extension_key: str = Header(default=""),
) -> dict[str, bool]:
    require_extension_key(x_extension_key)
    organization = find_organization_by_host(site_host)
    scope = scoped_client_id(client_id, site_host, employee_email)
    if not forget_fact(organization.id, scope, fact_id):
        raise HTTPException(status_code=404, detail="That remembered fact was not found.")
    return {"forgotten": True}


class SnapshotRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    client_id: str = Field(default="", validation_alias="clientId")
    site_host: str = Field(default="", validation_alias="siteHost")
    page_url: str = Field(default="", validation_alias="pageUrl")
    visible_text: str = Field(default="", validation_alias="visibleText")


@router.post("/extension/snapshots", response_model=SiteResponse, response_model_exclude_none=True)
def store_snapshot(
    body: SnapshotRequest,
    x_extension_key: str = Header(default=""),
) -> SiteResponse:
    require_extension_key(x_extension_key)
    if body.client_id.strip() == "":
        return SiteResponse(registered=False)
    try:
        site_host = normalize_site_host(body.site_host)
    except HTTPException:
        return SiteResponse(registered=False)
    registration = registration_for_host(site_host)
    if not registration.registered:
        return SiteResponse(registered=False)
    if not registration.portal_enabled:
        return SiteResponse(registered=True, portal_enabled=False)
    save_snapshot(body.client_id.strip(), site_host, body.page_url, body.visible_text)
    return SiteResponse(registered=True, portal_enabled=True)


@router.post("/extension/chat", response_model=ChatResponse)
def extension_chat(
    body: ChatRequest,
    x_extension_key: str = Header(default=""),
) -> ChatResponse:
    require_extension_key(x_extension_key)
    turn = take_chat_turn(
        body.client_id,
        body.site_host,
        body.message,
        body.session_id,
        body.answer_generally,
        employee_email=body.employee_email,
        answer_style=body.answer_style,
    )
    return chat_response(turn)


@router.post("/extension/chat/decline")
def extension_chat_decline(
    body: ChatRequest,
    x_extension_key: str = Header(default=""),
) -> dict[str, bool]:
    require_extension_key(x_extension_key)
    return decline_chat(body)


@router.post("/extension/chat/feedback", response_model=FeedbackResponse)
def extension_chat_feedback(
    body: FeedbackRequest,
    x_extension_key: str = Header(default=""),
) -> FeedbackResponse:
    require_extension_key(x_extension_key)
    return rate_answer(body)


@router.post("/extension/chat/reset", response_model=ResetResponse)
def extension_chat_reset(
    body: ChatRequest,
    x_extension_key: str = Header(default=""),
) -> ResetResponse:
    require_extension_key(x_extension_key)
    return reset_chat(body)


@router.post("/extension/chat/stop")
def extension_chat_stop(
    body: ChatRequest,
    x_extension_key: str = Header(default=""),
) -> dict[str, bool]:
    require_extension_key(x_extension_key)
    stop_turn(body.client_id)
    return {"stopped": True}


@router.post("/extension/chat/stream")
async def extension_chat_stream(
    body: ChatRequest,
    request: Request,
    x_extension_key: str = Header(default=""),
) -> StreamingResponse:
    require_extension_key(x_extension_key)
    return stream_chat(body, request)


@router.get("/extension/chat/{session_id}", response_model=HistoryResponse)
def extension_history(
    session_id: str,
    client_id: str = Query(default="", alias="clientId"),
    site_host: str = Query(default="", alias="siteHost"),
    employee_email: str = Query(default="", alias="employeeEmail"),
    x_extension_key: str = Header(default=""),
) -> HistoryResponse:
    require_extension_key(x_extension_key)
    return history_response(session_id, client_id, site_host, employee_email)
