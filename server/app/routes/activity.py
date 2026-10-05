from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel, Field

from app.activity.service import (
    ActivityRun,
    ActivityStep,
    RouteSpeed,
    list_activity,
    read_activity,
    read_live_steps,
    speed_summary,
)
from app.admin.session import require_admin_session, require_organization_access
from app.organizations.service import find_organization_by_host
from app.employees.records import scoped_client_id
from app.extension.access import require_extension_key
from app.organizations.service import normalize_site_host

router = APIRouter()


class ActivityRunResponse(BaseModel):
    id: str
    organization_id: str = Field(serialization_alias="organizationId")
    organization_name: str = Field(serialization_alias="organizationName")
    question: str
    standalone_question: str = Field(serialization_alias="standaloneQuestion")
    route: str
    plan_json: str = Field(serialization_alias="planJson")
    status: str
    total_ms: int = Field(serialization_alias="totalMs")
    first_token_ms: int | None = Field(serialization_alias="firstTokenMs")
    created_at: str = Field(serialization_alias="createdAt")
    rating: str
    feedback_note: str = Field(serialization_alias="feedbackNote")


class ActivityStepResponse(BaseModel):
    name: str
    status: str
    input_summary: str = Field(serialization_alias="inputSummary")
    output_summary: str = Field(serialization_alias="outputSummary")
    started_ms: int = Field(serialization_alias="startedMs")
    ended_ms: int = Field(serialization_alias="endedMs")


class ActivityDetailResponse(BaseModel):
    run: ActivityRunResponse
    steps: list[ActivityStepResponse]


class LiveActivityResponse(BaseModel):
    running: bool = False
    steps: list[ActivityStepResponse]


class RouteSpeedResponse(BaseModel):
    route: str
    turn_count: int = Field(serialization_alias="turnCount")
    median_total_ms: int | None = Field(serialization_alias="medianTotalMs")
    slowest_total_ms: int | None = Field(serialization_alias="slowestTotalMs")
    median_first_token_ms: int | None = Field(serialization_alias="medianFirstTokenMs")
    cache_hits: int = Field(serialization_alias="cacheHits")
    total_target_ms: int | None = Field(serialization_alias="totalTargetMs")
    first_token_target_ms: int | None = Field(serialization_alias="firstTokenTargetMs")
    baseline: str


def live_steps(client_id: str, site_host: str) -> LiveActivityResponse:
    cleaned_client = client_id.strip()
    if cleaned_client == "":
        return LiveActivityResponse()
    try:
        host = normalize_site_host(site_host)
    except HTTPException:
        return LiveActivityResponse()
    running, steps = read_live_steps(cleaned_client, host)
    return LiveActivityResponse(running=running, steps=[step_response(step) for step in steps])


def run_response(run: ActivityRun) -> ActivityRunResponse:
    return ActivityRunResponse(
        id=run.id,
        organization_id=run.organization_id,
        organization_name=run.organization_name,
        question=run.question,
        standalone_question=run.standalone_question,
        route=run.route,
        plan_json=run.plan_json,
        status=run.status,
        total_ms=run.total_ms,
        first_token_ms=run.first_token_ms,
        created_at=run.created_at,
        rating=run.rating,
        feedback_note=run.feedback_note,
    )


def step_response(step: ActivityStep) -> ActivityStepResponse:
    return ActivityStepResponse(
        name=step.name,
        status=step.status,
        input_summary=step.input_summary,
        output_summary=step.output_summary,
        started_ms=step.started_ms,
        ended_ms=step.ended_ms,
    )


def speed_response(speed: RouteSpeed) -> RouteSpeedResponse:
    return RouteSpeedResponse(
        route=speed.route,
        turn_count=speed.turn_count,
        median_total_ms=speed.median_total_ms,
        slowest_total_ms=speed.slowest_total_ms,
        median_first_token_ms=speed.median_first_token_ms,
        cache_hits=speed.cache_hits,
        total_target_ms=speed.total_target_ms,
        first_token_target_ms=speed.first_token_target_ms,
        baseline=speed.baseline,
    )


@router.get("/admin/activity", response_model=list[ActivityRunResponse])
def read_activity_list(
    rated_down: bool = Query(default=False, alias="ratedDown"),
    x_admin_session: str = Header(default=""),
) -> list[ActivityRunResponse]:
    principal = require_admin_session(x_admin_session)
    organization_id = "" if principal.is_platform_admin else principal.organization_id
    return [
        run_response(run)
        for run in list_activity(rated_down=rated_down, organization_id=organization_id)
    ]


@router.get("/admin/activity/speed", response_model=list[RouteSpeedResponse])
def read_speed_summary(x_admin_session: str = Header(default="")) -> list[RouteSpeedResponse]:
    principal = require_admin_session(x_admin_session)
    organization_id = "" if principal.is_platform_admin else principal.organization_id
    return [speed_response(speed) for speed in speed_summary(organization_id=organization_id)]


@router.get("/admin/activity/live", response_model=LiveActivityResponse)
def read_admin_live_activity(
    client_id: str = Query(default="", alias="clientId"),
    site_host: str = Query(default="", alias="siteHost"),
    x_admin_session: str = Header(default=""),
) -> LiveActivityResponse:
    organization = find_organization_by_host(site_host)
    require_organization_access(x_admin_session, organization.id)
    return live_steps(client_id, site_host)


@router.get("/extension/activity/live", response_model=LiveActivityResponse)
def read_extension_live_activity(
    client_id: str = Query(default="", alias="clientId"),
    site_host: str = Query(default="", alias="siteHost"),
    employee_email: str = Query(default="", alias="employeeEmail"),
    x_extension_key: str = Header(default=""),
) -> LiveActivityResponse:
    require_extension_key(x_extension_key)
    return live_steps(scoped_client_id(client_id, site_host, employee_email), site_host)


@router.get("/admin/activity/{run_id}", response_model=ActivityDetailResponse)
def read_activity_detail(
    run_id: str,
    x_admin_session: str = Header(default=""),
) -> ActivityDetailResponse:
    run, steps = read_activity(run_id)
    require_organization_access(x_admin_session, run.organization_id)
    return ActivityDetailResponse(run=run_response(run), steps=[step_response(step) for step in steps])
