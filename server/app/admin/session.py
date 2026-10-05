import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import bcrypt
from fastapi import HTTPException

from app.core.database import open_app_database
from app.core.settings import load_settings


@dataclass(frozen=True)
class SessionPrincipal:
    user_id: str
    email: str
    role: str
    organization_id: str

    @property
    def is_platform_admin(self) -> bool:
        return self.role == "platform_admin"


active_sessions: dict[str, SessionPrincipal] = {}


def password_matches(password: str, password_hash: str) -> bool:
    if password.strip() == "" or password_hash.strip() == "":
        return False
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def create_admin_session(principal: SessionPrincipal) -> str:
    token = secrets.token_urlsafe(32)
    active_sessions[token] = principal
    return token


def admin_session_is_active(token: str) -> bool:
    if token.strip() == "":
        return False
    return token in active_sessions


def end_admin_session(token: str) -> None:
    if token.strip() == "":
        return
    active_sessions.pop(token, None)


def authenticate_user(email: str, password: str) -> SessionPrincipal | None:
    normalized = email.strip().lower()
    settings = load_settings()
    if normalized == settings.admin_email.strip().lower() and password_matches(password, settings.admin_password_hash):
        return SessionPrincipal("platform-admin", normalized, "platform_admin", "")
    with open_app_database() as connection:
        row = connection.execute(
            """
            SELECT id, email, password_hash, role, COALESCE(organization_id, '') AS organization_id
            FROM admin_users WHERE lower(email) = ? AND active = 1
            """,
            (normalized,),
        ).fetchone()
    if row is None or not password_matches(password, str(row["password_hash"])):
        return None
    return SessionPrincipal(
        user_id=str(row["id"]),
        email=str(row["email"]),
        role=str(row["role"]),
        organization_id=str(row["organization_id"]),
    )


def create_organization_user(organization_id: str, email: str, password: str) -> None:
    normalized = email.strip().lower()
    if normalized == "" or "@" not in normalized:
        raise HTTPException(status_code=400, detail="Enter a valid email address.")
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Use a password with at least 8 characters.")
    password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    try:
        with open_app_database() as connection:
            connection.execute(
                """
                INSERT INTO admin_users (
                    id, email, password_hash, role, organization_id, active, created_at
                ) VALUES (?, ?, ?, 'organization_admin', ?, 1, ?)
                """,
                (str(uuid.uuid4()), normalized, password_hash, organization_id, datetime.now(timezone.utc).isoformat()),
            )
    except Exception as error:
        if "UNIQUE constraint failed" in str(error):
            raise HTTPException(status_code=409, detail="That email already has an account.") from error
        raise


def require_admin_session(token: str) -> SessionPrincipal:
    principal = active_sessions.get(token)
    if principal is None:
        raise HTTPException(status_code=401, detail="Sign in to continue.")
    return principal


def require_platform_admin(token: str) -> SessionPrincipal:
    principal = require_admin_session(token)
    if not principal.is_platform_admin:
        raise HTTPException(status_code=403, detail="A platform administrator is required.")
    return principal


def require_organization_access(token: str, organization_id: str) -> SessionPrincipal:
    principal = require_admin_session(token)
    if not principal.is_platform_admin and principal.organization_id != organization_id:
        raise HTTPException(status_code=403, detail="You do not have access to this organization.")
    return principal
