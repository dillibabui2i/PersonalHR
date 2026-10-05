from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from app.admin.session import create_organization_user, require_platform_admin
from app.core.database import open_app_database

router = APIRouter()


class OrganizationUserBody(BaseModel):
    email: str = ""
    password: str = ""


class OrganizationUserResponse(BaseModel):
    id: str
    email: str
    active: bool


@router.get(
    "/admin/organizations/{organization_id}/users",
    response_model=list[OrganizationUserResponse],
)
def read_organization_users(
    organization_id: str,
    x_admin_session: str = Header(default=""),
) -> list[OrganizationUserResponse]:
    require_platform_admin(x_admin_session)
    with open_app_database() as connection:
        rows = connection.execute(
            """
            SELECT id, email, active FROM admin_users
            WHERE organization_id = ? ORDER BY email
            """,
            (organization_id,),
        ).fetchall()
    return [
        OrganizationUserResponse(
            id=str(row["id"]),
            email=str(row["email"]),
            active=bool(row["active"]),
        )
        for row in rows
    ]


@router.post(
    "/admin/organizations/{organization_id}/users",
    response_model=dict[str, bool],
    status_code=201,
)
def add_organization_user(
    organization_id: str,
    body: OrganizationUserBody,
    x_admin_session: str = Header(default=""),
) -> dict[str, bool]:
    require_platform_admin(x_admin_session)
    create_organization_user(organization_id, body.email, body.password)
    return {"created": True}


@router.delete("/admin/organizations/{organization_id}/users/{user_id}")
def deactivate_organization_user(
    organization_id: str,
    user_id: str,
    x_admin_session: str = Header(default=""),
) -> dict[str, bool]:
    require_platform_admin(x_admin_session)
    with open_app_database() as connection:
        cursor = connection.execute(
            """
            UPDATE admin_users SET active = 0
            WHERE id = ? AND organization_id = ?
            """,
            (user_id, organization_id),
        )
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="That user was not found.")
    return {"deactivated": True}
