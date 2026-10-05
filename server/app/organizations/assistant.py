import re
import sqlite3
from dataclasses import dataclass
from urllib.parse import urlparse

from fastapi import HTTPException

from app.agent.planner import GREETING_REPLY, TurnPlan
from app.core.database import open_app_database
from app.organizations.service import read_organization

ANSWER_STYLES = {"brief", "detailed"}
GREETING_LIMIT = 400
PROFILE_PATH_LIMIT = 300
RELATIVE_PROFILE_PATH = re.compile(r"^/[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]*$")


@dataclass(frozen=True)
class AssistantSettings:
    organization_id: str
    greeting: str
    answer_style: str
    portal_enabled: bool
    memory_enabled: bool
    profile_api_path: str = ""


@dataclass(frozen=True)
class PlanLimits:
    portal_blocked: bool
    memory_blocked: bool


def default_settings(organization_id: str) -> AssistantSettings:
    return AssistantSettings(
        organization_id=organization_id,
        greeting=GREETING_REPLY,
        answer_style="brief",
        portal_enabled=True,
        memory_enabled=True,
        profile_api_path="",
    )


def row_text(row: sqlite3.Row, key: str, default: str = "") -> str:
    try:
        value = row[key]
    except (IndexError, KeyError):
        return default
    if value is None:
        return default
    return str(value).strip()


def settings_from_row(organization_id: str, row: sqlite3.Row) -> AssistantSettings:
    greeting = str(row["greeting"]).strip()
    answer_style = str(row["answer_style"])
    if greeting == "":
        greeting = GREETING_REPLY
    if answer_style not in ANSWER_STYLES:
        answer_style = "brief"
    return AssistantSettings(
        organization_id=organization_id,
        greeting=greeting,
        answer_style=answer_style,
        portal_enabled=bool(row["portal_enabled"]),
        memory_enabled=bool(row["memory_enabled"]),
        profile_api_path=row_text(row, "profile_api_path"),
    )


def read_assistant_settings(organization_id: str) -> AssistantSettings:
    read_organization(organization_id)
    with open_app_database() as connection:
        row = connection.execute(
            """
            SELECT greeting, answer_style, portal_enabled, memory_enabled, profile_api_path
            FROM organization_settings
            WHERE organization_id = ?
            """,
            (organization_id,),
        ).fetchone()
    if row is None:
        return default_settings(organization_id)
    return settings_from_row(organization_id, row)


def clean_profile_api_path(value: str) -> str:
    cleaned = value.strip()
    if cleaned == "":
        return ""
    if len(cleaned) > PROFILE_PATH_LIMIT:
        raise HTTPException(status_code=400, detail="Keep the profile API path under 300 characters.")
    if cleaned.startswith("/"):
        if RELATIVE_PROFILE_PATH.fullmatch(cleaned) is None:
            raise HTTPException(status_code=400, detail="Enter a valid profile API path starting with /.")
        return cleaned
    parsed = urlparse(cleaned)
    if parsed.scheme not in {"http", "https"} or parsed.netloc == "" or parsed.path == "":
        raise HTTPException(
            status_code=400,
            detail="Enter a relative path like /api/me or a full http(s) URL on the portal host.",
        )
    return cleaned


def save_assistant_settings(
    organization_id: str,
    greeting: str,
    answer_style: str,
    portal_enabled: bool,
    memory_enabled: bool,
    profile_api_path: str = "",
) -> AssistantSettings:
    read_organization(organization_id)
    cleaned_greeting = " ".join(greeting.split()).strip()
    if cleaned_greeting == "":
        raise HTTPException(status_code=400, detail="Enter a greeting.")
    if len(cleaned_greeting) > GREETING_LIMIT:
        raise HTTPException(status_code=400, detail="Keep the greeting under 400 characters.")
    if answer_style not in ANSWER_STYLES:
        raise HTTPException(status_code=400, detail="Choose Brief or Detailed.")
    api_path = clean_profile_api_path(profile_api_path)
    with open_app_database() as connection:
        connection.execute(
            """
            INSERT INTO organization_settings (
                organization_id, greeting, answer_style, portal_enabled, memory_enabled, starter_questions_json,
                profile_api_path
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(organization_id) DO UPDATE SET
                greeting = excluded.greeting,
                answer_style = excluded.answer_style,
                portal_enabled = excluded.portal_enabled,
                memory_enabled = excluded.memory_enabled,
                starter_questions_json = excluded.starter_questions_json,
                profile_api_path = excluded.profile_api_path
            """,
            (
                organization_id,
                cleaned_greeting,
                answer_style,
                1 if portal_enabled else 0,
                1 if memory_enabled else 0,
                "[]",
                api_path,
            ),
        )
    return AssistantSettings(
        organization_id=organization_id,
        greeting=cleaned_greeting,
        answer_style=answer_style,
        portal_enabled=portal_enabled,
        memory_enabled=memory_enabled,
        profile_api_path=api_path,
    )


def limit_plan(plan: TurnPlan, settings: AssistantSettings) -> PlanLimits:
    """Turn off tools this organization disabled. A page-only or memory-only turn is blocked."""
    portal_only = (
        plan.needs_portal
        and not plan.needs_policy
        and not plan.needs_dates
        and not plan.needs_memory
        and plan.stated_facts == []
    )
    memory_only = (plan.needs_memory or plan.stated_facts != []) and not plan.needs_policy and not plan.needs_portal and not plan.needs_dates
    if not settings.portal_enabled:
        plan.needs_portal = False
    if not settings.memory_enabled:
        plan.needs_memory = False
        plan.stated_facts = []
    return PlanLimits(
        portal_blocked=not settings.portal_enabled and portal_only,
        memory_blocked=not settings.memory_enabled and memory_only,
    )
