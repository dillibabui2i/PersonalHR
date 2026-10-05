import re
import shutil
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse

from fastapi import HTTPException

from app.core.database import data_directory, open_app_database

HOST_PATTERN = re.compile(r"^(?=.{1,253}$)([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,}$")


@dataclass(frozen=True)
class OrganizationRecord:
    id: str
    name: str
    description: str
    site_host: str


def normalize_site_host(value: str) -> str:
    trimmed = value.strip()
    if trimmed == "":
        raise HTTPException(status_code=400, detail="Enter a site host.")
    if any(character.isspace() for character in trimmed):
        raise HTTPException(
            status_code=400,
            detail="Enter a full site host, such as portal.example.com.",
        )
    parsed = urlparse(trimmed if "://" in trimmed else f"//{trimmed}")
    host = (parsed.hostname or "").lower().rstrip(".")
    if HOST_PATTERN.fullmatch(host) is None:
        raise HTTPException(
            status_code=400,
            detail="Enter a full site host, such as portal.example.com.",
        )
    return host


def organization_from_row(row: sqlite3.Row) -> OrganizationRecord:
    return OrganizationRecord(
        id=row["id"],
        name=row["name"],
        description=row["description"],
        site_host=row["site_host"],
    )


def list_organizations() -> list[OrganizationRecord]:
    with open_app_database() as connection:
        rows = connection.execute(
            """
            SELECT id, name, description, site_host
            FROM organizations
            ORDER BY created_at
            """
        ).fetchall()
    return [organization_from_row(row) for row in rows]


def organization_for_saved_host(site_host: str) -> OrganizationRecord | None:
    with open_app_database() as connection:
        row = connection.execute(
            """
            SELECT id, name, description, site_host
            FROM organizations
            WHERE site_host = ?
            """,
            (site_host,),
        ).fetchone()
    if row is None:
        return None
    return organization_from_row(row)


def find_organization_by_host(site_host: str) -> OrganizationRecord:
    organization = organization_for_saved_host(normalize_site_host(site_host))
    if organization is None:
        raise HTTPException(status_code=404, detail="That site is not registered.")
    return organization


def read_organization(organization_id: str) -> OrganizationRecord:
    with open_app_database() as connection:
        row = connection.execute(
            """
            SELECT id, name, description, site_host
            FROM organizations
            WHERE id = ?
            """,
            (organization_id,),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="That organization was not found.")
    return organization_from_row(row)


def update_organization(
    organization_id: str,
    name: str,
    description: str,
    site_host: str,
) -> OrganizationRecord:
    organization = read_organization(organization_id)
    cleaned_name = name.strip()
    if cleaned_name == "":
        raise HTTPException(status_code=400, detail="Enter an organization name.")
    cleaned_description = description.strip()
    cleaned_host = normalize_site_host(site_host)
    if (
        cleaned_name == organization.name
        and cleaned_description == organization.description
        and cleaned_host == organization.site_host
    ):
        return organization
    try:
        with open_app_database() as connection:
            connection.execute(
                """
                UPDATE organizations
                SET name = ?, description = ?, site_host = ?
                WHERE id = ?
                """,
                (cleaned_name, cleaned_description, cleaned_host, organization_id),
            )
    except sqlite3.IntegrityError:
        raise HTTPException(
            status_code=409,
            detail="An organization already uses that site host.",
        ) from None
    return OrganizationRecord(
        id=organization.id,
        name=cleaned_name,
        description=cleaned_description,
        site_host=cleaned_host,
    )


def create_organization(
    name: str,
    description: str,
    site_host: str,
    embed_model: str = "",
) -> OrganizationRecord:
    cleaned_name = name.strip()
    if cleaned_name == "":
        raise HTTPException(status_code=400, detail="Enter an organization name.")
    cleaned_description = description.strip()
    cleaned_host = normalize_site_host(site_host)
    organization = OrganizationRecord(
        id=str(uuid.uuid4()),
        name=cleaned_name,
        description=cleaned_description,
        site_host=cleaned_host,
    )
    created_at = datetime.now(timezone.utc).isoformat()
    try:
        with open_app_database() as connection:
            connection.execute(
                """
                INSERT INTO organizations (id, name, description, site_host, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    organization.id,
                    organization.name,
                    organization.description,
                    organization.site_host,
                    created_at,
                ),
            )
    except sqlite3.IntegrityError:
        raise HTTPException(
            status_code=409,
            detail="An organization already uses that site host.",
        ) from None
    if embed_model.strip() == "":
        return organization
    from app.organizations.models import lock_embedding_model

    try:
        lock_embedding_model(organization.id, embed_model)
    except HTTPException:
        with open_app_database() as connection:
            connection.execute("DELETE FROM organizations WHERE id = ?", (organization.id,))
        raise
    return organization


def delete_organization(organization_id: str) -> None:
    from app.agent.persistence import delete_flow_sessions

    organization = read_organization(organization_id)
    with open_app_database() as connection:
        session_rows = connection.execute(
            "SELECT id FROM chat_sessions WHERE organization_id = ?",
            (organization.id,),
        ).fetchall()
        delete_flow_sessions([str(row["id"]) for row in session_rows])
        connection.execute(
            """
            DELETE FROM answer_feedback
            WHERE run_id IN (SELECT id FROM agent_runs WHERE organization_id = ?)
            """,
            (organization.id,),
        )
        connection.execute(
            """
            DELETE FROM chat_messages
            WHERE session_id IN (SELECT id FROM chat_sessions WHERE organization_id = ?)
            """,
            (organization.id,),
        )
        connection.execute("DELETE FROM chat_sessions WHERE organization_id = ?", (organization.id,))
        connection.execute("DELETE FROM portal_snapshots WHERE site_host = ?", (organization.site_host,))
        connection.execute("DELETE FROM portal_sessions WHERE site_host = ?", (organization.site_host,))
        connection.execute("DELETE FROM documents WHERE organization_id = ?", (organization.id,))
        connection.execute(
            """
            DELETE FROM agent_steps
            WHERE run_id IN (SELECT id FROM agent_runs WHERE organization_id = ?)
            """,
            (organization.id,),
        )
        connection.execute("DELETE FROM agent_runs WHERE organization_id = ?", (organization.id,))
        connection.execute(
            """
            DELETE FROM quality_results
            WHERE check_id IN (SELECT id FROM quality_checks WHERE organization_id = ?)
            """,
            (organization.id,),
        )
        connection.execute("DELETE FROM quality_checks WHERE organization_id = ?", (organization.id,))
        connection.execute("DELETE FROM quality_runs WHERE organization_id = ?", (organization.id,))
        connection.execute("DELETE FROM employee_facts WHERE organization_id = ?", (organization.id,))
        connection.execute("DELETE FROM employees WHERE organization_id = ?", (organization.id,))
        connection.execute("DELETE FROM answer_cache WHERE organization_id = ?", (organization.id,))
        connection.execute("DELETE FROM model_switch_requests WHERE organization_id = ?", (organization.id,))
        connection.execute("DELETE FROM organization_models WHERE organization_id = ?", (organization.id,))
        connection.execute("DELETE FROM admin_users WHERE organization_id = ?", (organization.id,))
        connection.execute("DELETE FROM organization_settings WHERE organization_id = ?", (organization.id,))
        connection.execute("DELETE FROM organizations WHERE id = ?", (organization.id,))
    file_directory = data_directory() / "files" / organization.id
    if file_directory.exists():
        shutil.rmtree(file_directory)
    vector_file = data_directory() / "vectors" / f"{uuid.UUID(organization.id)}.db"
    for suffix in ("", "-wal", "-shm"):
        candidate = vector_file.parent / f"{vector_file.name}{suffix}"
        if candidate.is_file():
            candidate.unlink()
