import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from fastapi import HTTPException

from app.core.database import open_app_database
from app.core.settings import load_settings


@dataclass(frozen=True)
class ModelSwitchRequest:
    id: str
    organization_id: str
    organization_name: str
    requested_model: str
    reason: str
    status: str
    requested_by: str
    review_note: str
    created_at: str


def available_chat_models() -> list[str]:
    from app.llm.catalog import installed_by_kind

    installed = installed_by_kind("chat")
    if installed != []:
        return installed
    return [load_settings().chat_model]


def available_embed_models() -> list[dict[str, object]]:
    from app.llm.catalog import installed_by_kind, known_dimensions

    models: list[dict[str, object]] = []
    for name in installed_by_kind("embed"):
        models.append({"name": name, "dimensions": known_dimensions(name)})
    return models


def _model_row(organization_id: str) -> tuple[str, str, int] | None:
    with open_app_database() as connection:
        row = connection.execute(
            """
            SELECT chat_model, embed_model, embed_dimensions
            FROM organization_models WHERE organization_id = ?
            """,
            (organization_id,),
        ).fetchone()
    if row is None:
        return None
    return str(row["chat_model"]), str(row["embed_model"]), int(row["embed_dimensions"])


def ensure_organization_model(organization_id: str) -> tuple[str, str, int]:
    """Keep a legacy organization on the server embedding model. That choice stays fixed."""
    settings = load_settings()
    current = _model_row(organization_id)
    now = datetime.now(timezone.utc).isoformat()
    if current is None:
        with open_app_database() as connection:
            connection.execute(
                """
                INSERT INTO organization_models (
                    organization_id, chat_model, updated_at, embed_model, embed_dimensions
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (organization_id, settings.chat_model, now, settings.embed_model, settings.embed_dimensions),
            )
        return settings.chat_model, settings.embed_model, settings.embed_dimensions
    chat_model, embed_model, dimensions = current
    if embed_model != "" and dimensions > 0:
        return chat_model, embed_model, dimensions
    embed_model = settings.embed_model
    dimensions = settings.embed_dimensions
    with open_app_database() as connection:
        connection.execute(
            """
            UPDATE organization_models
            SET embed_model = ?, embed_dimensions = ?, updated_at = ?
            WHERE organization_id = ?
            """,
            (embed_model, dimensions, now, organization_id),
        )
    return chat_model, embed_model, dimensions


def organization_chat_model(organization_id: str) -> str:
    chat_model, _embed_model, _dimensions = ensure_organization_model(organization_id)
    return chat_model


def organization_embedding(organization_id: str) -> tuple[str, int]:
    _chat_model, embed_model, dimensions = ensure_organization_model(organization_id)
    return embed_model, dimensions


def embedding_dimensions(model: str) -> int:
    from app.llm.catalog import known_dimensions
    from app.llm.embeddings import EmbeddingFailed, probe_dimensions

    known = known_dimensions(model)
    if known > 0:
        return known
    try:
        dimensions = probe_dimensions(model)
    except EmbeddingFailed as error:
        raise HTTPException(status_code=400, detail=error.detail) from error
    from app.llm.catalog import remember_kind

    remember_kind(model, "embed", dimensions)
    return dimensions


def lock_embedding_model(organization_id: str, model: str) -> tuple[str, int]:
    """Choose the embedding model once. Later document search depends on this vector size."""
    from app.llm.catalog import installed_by_kind
    from app.llm.model_status import model_is_installed as name_is_installed

    settings = load_settings()
    chosen = model.strip() or settings.embed_model
    if not name_is_installed(installed_by_kind("embed"), chosen):
        raise HTTPException(
            status_code=400,
            detail="Install that embedding model before selecting it for an organization.",
        )
    current = _model_row(organization_id)
    if current is not None and current[1] != "":
        raise HTTPException(
            status_code=409,
            detail="The embedding model for this organization cannot be changed.",
        )
    dimensions = embedding_dimensions(chosen)
    chat_model = settings.chat_model if current is None else current[0]
    now = datetime.now(timezone.utc).isoformat()
    with open_app_database() as connection:
        connection.execute(
            """
            INSERT INTO organization_models (
                organization_id, chat_model, updated_at, embed_model, embed_dimensions
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(organization_id) DO UPDATE SET
                embed_model = excluded.embed_model,
                embed_dimensions = excluded.embed_dimensions,
                updated_at = excluded.updated_at
            """,
            (organization_id, chat_model, now, chosen, dimensions),
        )
    return chosen, dimensions


def setup_organization_models(organization_id: str, chat_model: str, embed_model: str) -> tuple[str, str]:
    """Choose the chat model and lock the embedding model while the organization is created."""
    chosen_embed = embed_model.strip()
    if chosen_embed == "":
        raise HTTPException(status_code=400, detail="Choose an embedding model.")
    current = _model_row(organization_id)
    if current is None or current[1] == "":
        chosen_embed, _dimensions = lock_embedding_model(organization_id, chosen_embed)
    elif current[1] != chosen_embed:
        raise HTTPException(
            status_code=409,
            detail="The embedding model for this organization cannot be changed.",
        )
    return assign_chat_model(organization_id, chat_model), chosen_embed


def assign_chat_model(organization_id: str, model: str) -> str:
    cleaned = model.strip()
    if cleaned == "":
        raise HTTPException(status_code=400, detail="Choose a chat model.")
    if cleaned not in available_chat_models():
        raise HTTPException(status_code=400, detail="That chat model is not installed.")
    ensure_organization_model(organization_id)
    with open_app_database() as connection:
        connection.execute(
            """
            UPDATE organization_models
            SET chat_model = ?, updated_at = ?
            WHERE organization_id = ?
            """,
            (cleaned, datetime.now(timezone.utc).isoformat(), organization_id),
        )
    return cleaned


def request_model_switch(organization_id: str, model: str, reason: str, requested_by: str) -> str:
    cleaned = model.strip()
    if cleaned == "":
        raise HTTPException(status_code=400, detail="Choose a requested model.")
    request_id = str(uuid.uuid4())
    with open_app_database() as connection:
        connection.execute(
            """
            INSERT INTO model_switch_requests (
                id, organization_id, requested_model, reason, status, requested_by,
                reviewed_by, review_note, created_at, reviewed_at
            ) VALUES (?, ?, ?, ?, 'pending', ?, '', '', ?, '')
            """,
            (
                request_id,
                organization_id,
                cleaned,
                reason.strip(),
                requested_by,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
    return request_id


def list_model_requests(organization_id: str = "") -> list[ModelSwitchRequest]:
    with open_app_database() as connection:
        rows = connection.execute(
            """
            SELECT model_switch_requests.*, organizations.name AS organization_name
            FROM model_switch_requests
            JOIN organizations ON organizations.id = model_switch_requests.organization_id
            WHERE (? = '' OR model_switch_requests.organization_id = ?)
            ORDER BY model_switch_requests.created_at DESC
            """,
            (organization_id, organization_id),
        ).fetchall()
    return [
        ModelSwitchRequest(
            id=str(row["id"]),
            organization_id=str(row["organization_id"]),
            organization_name=str(row["organization_name"]),
            requested_model=str(row["requested_model"]),
            reason=str(row["reason"]),
            status=str(row["status"]),
            requested_by=str(row["requested_by"]),
            review_note=str(row["review_note"]),
            created_at=str(row["created_at"]),
        )
        for row in rows
    ]


def review_model_request(
    request_id: str,
    decision: str,
    note: str,
    reviewed_by: str,
) -> ModelSwitchRequest:
    if decision not in {"approved", "rejected"}:
        raise HTTPException(status_code=400, detail="Choose approved or rejected.")
    with open_app_database() as connection:
        row = connection.execute(
            "SELECT organization_id, requested_model, status FROM model_switch_requests WHERE id = ?",
            (request_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="That request was not found.")
        if str(row["status"]) != "pending":
            raise HTTPException(status_code=409, detail="That request has already been reviewed.")
    if decision == "approved":
        assign_chat_model(str(row["organization_id"]), str(row["requested_model"]))
    with open_app_database() as connection:
        connection.execute(
            """
            UPDATE model_switch_requests
            SET status = ?, reviewed_by = ?, review_note = ?, reviewed_at = ?
            WHERE id = ?
            """,
            (decision, reviewed_by, note.strip(), datetime.now(timezone.utc).isoformat(), request_id),
        )
    return next(item for item in list_model_requests() if item.id == request_id)
