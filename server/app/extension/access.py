import secrets

from fastapi import HTTPException

from app.core.settings import load_settings


def require_extension_key(token: str) -> None:
    expected = load_settings().extension_key.strip()
    provided = token.strip()
    if expected == "" or provided == "" or not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="The extension key was not accepted.")
