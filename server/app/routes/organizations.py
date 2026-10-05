from fastapi import APIRouter, Header, Response
from pydantic import BaseModel, ConfigDict, Field

from app.admin.session import (
    require_admin_session,
    require_organization_access,
    require_platform_admin,
)
from app.organizations.assistant import (
    AssistantSettings,
    read_assistant_settings,
    save_assistant_settings,
)
from app.employees.records import EmployeeRecord, list_employees
from app.organizations.service import (
    OrganizationRecord,
    create_organization,
    delete_organization,
    list_organizations,
    read_organization,
    update_organization,
)

router = APIRouter()


class OrganizationBody(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str = ""
    description: str = ""
    site_host: str = Field(default="", validation_alias="siteHost")
    embed_model: str = Field(default="", validation_alias="embedModel")


class OrganizationResponse(BaseModel):
    id: str
    name: str
    description: str
    site_host: str = Field(serialization_alias="siteHost")


class EmployeeResponse(BaseModel):
    id: str
    email: str
    display_name: str = Field(serialization_alias="displayName")
    employee_code: str = Field(serialization_alias="employeeCode")
    department: str
    role_title: str = Field(serialization_alias="roleTitle")
    joining_date: str = Field(serialization_alias="joiningDate")
    work_location: str = Field(serialization_alias="workLocation")
    onboarded: bool
    updated_at: str = Field(serialization_alias="updatedAt")


def employee_response(employee: EmployeeRecord) -> EmployeeResponse:
    return EmployeeResponse(
        id=employee.id,
        email=employee.email,
        display_name=employee.display_name,
        employee_code=employee.employee_code,
        department=employee.department,
        role_title=employee.role_title,
        joining_date=employee.joining_date,
        work_location=employee.work_location,
        onboarded=employee.onboarded_at != "",
        updated_at=employee.updated_at,
    )


def organization_response(organization: OrganizationRecord) -> OrganizationResponse:
    return OrganizationResponse(
        id=organization.id,
        name=organization.name,
        description=organization.description,
        site_host=organization.site_host,
    )


@router.get("/admin/organizations", response_model=list[OrganizationResponse])
def read_organizations(x_admin_session: str = Header(default="")) -> list[OrganizationResponse]:
    principal = require_admin_session(x_admin_session)
    organizations = list_organizations()
    if not principal.is_platform_admin:
        organizations = [item for item in organizations if item.id == principal.organization_id]
    return [organization_response(organization) for organization in organizations]


@router.post("/admin/organizations", response_model=OrganizationResponse, status_code=201)
def add_organization(
    body: OrganizationBody,
    x_admin_session: str = Header(default=""),
) -> OrganizationResponse:
    require_platform_admin(x_admin_session)
    organization = create_organization(body.name, body.description, body.site_host, body.embed_model)
    return organization_response(organization)


@router.get("/admin/organizations/{organization_id}/employees", response_model=list[EmployeeResponse])
def read_employees(
    organization_id: str,
    x_admin_session: str = Header(default=""),
) -> list[EmployeeResponse]:
    require_organization_access(x_admin_session, organization_id)
    return [employee_response(employee) for employee in list_employees(organization_id)]


@router.get("/admin/organizations/{organization_id}", response_model=OrganizationResponse)
def read_organization_route(
    organization_id: str,
    x_admin_session: str = Header(default=""),
) -> OrganizationResponse:
    require_organization_access(x_admin_session, organization_id)
    return organization_response(read_organization(organization_id))


@router.patch("/admin/organizations/{organization_id}", response_model=OrganizationResponse)
def update_organization_route(
    organization_id: str,
    body: OrganizationBody,
    x_admin_session: str = Header(default=""),
) -> OrganizationResponse:
    require_organization_access(x_admin_session, organization_id)
    organization = update_organization(organization_id, body.name, body.description, body.site_host)
    return organization_response(organization)


@router.delete("/admin/organizations/{organization_id}", status_code=204)
def remove_organization(
    organization_id: str,
    x_admin_session: str = Header(default=""),
) -> Response:
    require_platform_admin(x_admin_session)
    delete_organization(organization_id)
    return Response(status_code=204)


class AssistantSettingsBody(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    greeting: str = ""
    answer_style: str = Field(default="brief", validation_alias="answerStyle")
    portal_enabled: bool = Field(default=True, validation_alias="portalEnabled")
    memory_enabled: bool = Field(default=True, validation_alias="memoryEnabled")
    profile_api_path: str = Field(default="", validation_alias="profileApiPath")


class AssistantSettingsResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    greeting: str
    answer_style: str = Field(serialization_alias="answerStyle")
    portal_enabled: bool = Field(serialization_alias="portalEnabled")
    memory_enabled: bool = Field(serialization_alias="memoryEnabled")
    profile_api_path: str = Field(serialization_alias="profileApiPath")


def assistant_settings_response(settings: AssistantSettings) -> AssistantSettingsResponse:
    return AssistantSettingsResponse(
        greeting=settings.greeting,
        answer_style=settings.answer_style,
        portal_enabled=settings.portal_enabled,
        memory_enabled=settings.memory_enabled,
        profile_api_path=settings.profile_api_path,
    )


@router.get(
    "/admin/organizations/{organization_id}/assistant-settings",
    response_model=AssistantSettingsResponse,
)
def read_assistant_settings_route(
    organization_id: str,
    x_admin_session: str = Header(default=""),
) -> AssistantSettingsResponse:
    require_organization_access(x_admin_session, organization_id)
    return assistant_settings_response(read_assistant_settings(organization_id))


@router.put(
    "/admin/organizations/{organization_id}/assistant-settings",
    response_model=AssistantSettingsResponse,
)
def save_assistant_settings_route(
    organization_id: str,
    body: AssistantSettingsBody,
    x_admin_session: str = Header(default=""),
) -> AssistantSettingsResponse:
    require_organization_access(x_admin_session, organization_id)
    settings = save_assistant_settings(
        organization_id,
        body.greeting,
        body.answer_style,
        body.portal_enabled,
        body.memory_enabled,
        body.profile_api_path,
    )
    return assistant_settings_response(settings)
