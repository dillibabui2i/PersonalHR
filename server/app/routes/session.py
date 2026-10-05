from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from app.admin.session import (
    authenticate_user,
    create_admin_session,
    end_admin_session,
    require_admin_session,
)

router = APIRouter()


class SessionRequest(BaseModel):
    email: str = ""
    password: str = Field(min_length=0)


class SessionResponse(BaseModel):
    sessionToken: str
    role: str
    organizationId: str
    email: str


class SignedInResponse(BaseModel):
    signedIn: bool
    role: str = ""
    organizationId: str = ""
    email: str = ""


@router.post("/admin/session", response_model=SessionResponse)
def open_admin_session(body: SessionRequest) -> SessionResponse:
    if body.email.strip() == "":
        raise HTTPException(status_code=400, detail="Enter your email address.")
    if body.password.strip() == "":
        raise HTTPException(status_code=400, detail="Enter your password.")
    principal = authenticate_user(body.email, body.password)
    if principal is None:
        raise HTTPException(status_code=401, detail="The email or password is not correct.")
    return SessionResponse(
        sessionToken=create_admin_session(principal),
        role=principal.role,
        organizationId=principal.organization_id,
        email=principal.email,
    )


@router.get("/admin/session", response_model=SignedInResponse)
def read_admin_session(x_admin_session: str = Header(default="")) -> SignedInResponse:
    principal = require_admin_session(x_admin_session)
    return SignedInResponse(
        signedIn=True,
        role=principal.role,
        organizationId=principal.organization_id,
        email=principal.email,
    )


@router.delete("/admin/session", response_model=SignedInResponse)
def close_admin_session(x_admin_session: str = Header(default="")) -> SignedInResponse:
    end_admin_session(x_admin_session)
    return SignedInResponse(signedIn=False)
