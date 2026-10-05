from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.admin.session import (
    require_admin_session,
    require_organization_access,
    require_platform_admin,
)

from app.llm.catalog import catalogue_view, ollama_reachable, pull_snapshot, start_pull
from app.organizations.models import (
    ModelSwitchRequest,
    assign_chat_model,
    available_chat_models,
    available_embed_models,
    list_model_requests,
    organization_chat_model,
    organization_embedding,
    request_model_switch,
    review_model_request,
    setup_organization_models,
)

router = APIRouter()


class ModelAssignmentBody(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    model: str = ""
    embed_model: str = Field(default="", validation_alias="embedModel")


class ModelPullBody(BaseModel):
    name: str = ""
    kind: str = "chat"


class ModelRequestBody(BaseModel):
    model: str = ""
    reason: str = ""


class ModelReviewBody(BaseModel):
    decision: str = ""
    note: str = ""


class ModelRequestResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    organization_id: str = Field(serialization_alias="organizationId")
    organization_name: str = Field(serialization_alias="organizationName")
    requested_model: str = Field(serialization_alias="requestedModel")
    reason: str
    status: str
    requested_by: str = Field(serialization_alias="requestedBy")
    review_note: str = Field(serialization_alias="reviewNote")
    created_at: str = Field(serialization_alias="createdAt")


def request_response(item: ModelSwitchRequest) -> ModelRequestResponse:
    return ModelRequestResponse(
        id=item.id,
        organization_id=item.organization_id,
        organization_name=item.organization_name,
        requested_model=item.requested_model,
        reason=item.reason,
        status=item.status,
        requested_by=item.requested_by,
        review_note=item.review_note,
        created_at=item.created_at,
    )


@router.get("/admin/models/catalog")
def model_catalog(x_admin_session: str = Header(default="")) -> dict[str, object]:
    require_admin_session(x_admin_session)
    return {
        "ollamaReachable": ollama_reachable(),
        "installedChat": available_chat_models(),
        "installedEmbed": available_embed_models(),
        "catalogue": catalogue_view(),
        "pulls": list(pull_snapshot().values()),
    }


@router.post("/admin/models/pull", status_code=202)
def pull_model(
    body: ModelPullBody,
    x_admin_session: str = Header(default=""),
) -> dict[str, str]:
    require_platform_admin(x_admin_session)
    try:
        start_pull(body.name, body.kind)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"name": body.name.strip(), "status": "pulling"}


@router.get("/admin/organizations/{organization_id}/model")
def read_model_assignment(
    organization_id: str,
    x_admin_session: str = Header(default=""),
) -> dict[str, object]:
    require_organization_access(x_admin_session, organization_id)
    embed_model, dimensions = organization_embedding(organization_id)
    return {
        "model": organization_chat_model(organization_id),
        "embedModel": embed_model,
        "embedDimensions": dimensions,
    }


@router.put("/admin/organizations/{organization_id}/model")
def save_model_assignment(
    organization_id: str,
    body: ModelAssignmentBody,
    x_admin_session: str = Header(default=""),
) -> dict[str, str]:
    require_platform_admin(x_admin_session)
    if body.embed_model.strip() != "":
        chat_model, embed_model = setup_organization_models(
            organization_id,
            body.model,
            body.embed_model,
        )
        return {"model": chat_model, "embedModel": embed_model}
    return {"model": assign_chat_model(organization_id, body.model)}


@router.post("/admin/organizations/{organization_id}/model-requests", status_code=201)
def create_model_request(
    organization_id: str,
    body: ModelRequestBody,
    x_admin_session: str = Header(default=""),
) -> dict[str, str]:
    principal = require_organization_access(x_admin_session, organization_id)
    return {
        "id": request_model_switch(
            organization_id,
            body.model,
            body.reason,
            principal.email,
        )
    }


@router.get("/admin/model-requests", response_model=list[ModelRequestResponse])
def read_model_requests(
    x_admin_session: str = Header(default=""),
) -> list[ModelRequestResponse]:
    principal = require_admin_session(x_admin_session)
    organization_id = "" if principal.is_platform_admin else principal.organization_id
    return [request_response(item) for item in list_model_requests(organization_id)]


@router.patch("/admin/model-requests/{request_id}", response_model=ModelRequestResponse)
def process_model_request(
    request_id: str,
    body: ModelReviewBody,
    x_admin_session: str = Header(default=""),
) -> ModelRequestResponse:
    principal = require_platform_admin(x_admin_session)
    return request_response(
        review_model_request(request_id, body.decision, body.note, principal.email)
    )
