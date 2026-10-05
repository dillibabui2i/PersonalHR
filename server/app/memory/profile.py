import re
from dataclasses import dataclass
from typing import Any

from app.core.settings import load_settings
from app.llm.chat_model import ChatModelFailed, generate_json
from app.employees.records import normalize_email
from app.memory.service import SECRET, clean_fact

SECRET_KEY = re.compile(
    r"(password|passcode|secret|token|otp|ssn|pan|aadhaar|bank|account.?number|salary|ctc|compensation)",
    re.IGNORECASE,
)
PLACEHOLDER = re.compile(
    r"(?i)(null|none|nil|n/?a|-+|unknown|tbd|not (?:specified|available|provided|present|mentioned|listed|found))"
)
PROFILE_LIMIT = 3500
FACT_LIMIT = 8
LIST_LIMIT = 3
VALUE_LIMIT = 160
OUTPUT_TOKENS = 320
NOISE_PATH = re.compile(
    r"skill|version|certification|experience|leave|balance|attendance|timesheet|notification|permission|avatar|image|photo|project",
    re.IGNORECASE,
)

PROFILE_PROMPT = """Extract this employee's identification for a private HR assistant.

The profile below is "path: value" lines. Return only who this employee is.

- displayName: the employee's own name. Never a job title, email, or code. Use "" if no name is present.
- facts: at most {limit} items. Each item is a short label and a value copied from the lines.
  Keep only these, and use these labels: Work email, Employee code, Role, Department, Manager, Work location, Joining date.
- Skip skills, skill versions, experience, projects, leave, balances, and details about other people.
- Do not invent a value. If it is not in the lines, leave it out.

Profile:
{profile}
"""

PROFILE_SCHEMA = {
    "type": "object",
    "properties": {
        "displayName": {"type": "string"},
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["label", "value"],
            },
            "maxItems": FACT_LIMIT,
        },
    },
    "required": ["displayName", "facts"],
}


class ProfileExtractionFailed(Exception):
    pass


def email_key(key: str) -> bool:
    cleaned = re.sub(r"[^a-z]", "", key.lower())
    return cleaned.endswith(("email", "emailaddress", "emailid", "mail"))


def profile_email(payload: Any, depth: int = 0) -> str:
    """Read the employee's email from the organization's profile response."""
    found: list[tuple[int, str]] = []

    def walk(value: Any, key: str, level: int) -> None:
        if level > 6:
            return
        if isinstance(value, dict):
            for child_key, nested in value.items():
                walk(nested, str(child_key), level + 1)
            return
        if isinstance(value, list):
            for item in value[:20]:
                walk(item, key, level + 1)
            return
        if not isinstance(value, str) or not email_key(key):
            return
        email = normalize_email(value)
        if email == "":
            match = re.search(r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}", value, re.IGNORECASE)
            email = normalize_email(match.group(0)) if match is not None else ""
        if email != "":
            found.append((level, email))

    walk(payload, "", depth)
    if found == []:
        return ""
    found.sort(key=lambda item: item[0])
    return found[0][1]


@dataclass(frozen=True)
class ProfileMemory:
    display_name: str
    facts: list[str]


def redact_profile(value: Any, depth: int = 0) -> Any:
    if depth > 6:
        return None
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, nested in value.items():
            if SECRET_KEY.search(str(key)) is not None:
                continue
            redacted = redact_profile(nested, depth + 1)
            if redacted is None or redacted == "" or redacted == [] or redacted == {}:
                continue
            cleaned[str(key)] = redacted
        return cleaned
    if isinstance(value, list):
        items = [redact_profile(item, depth + 1) for item in value[:20]]
        return [item for item in items if item not in (None, "", [], {})]
    if isinstance(value, str):
        text = " ".join(value.split()).strip()
        if text == "" or SECRET.search(text) is not None:
            return ""
        return text[:240]
    if isinstance(value, (int, float, bool)):
        return value
    return None


def profile_lines(value: Any, path: str = "", depth: int = 0) -> list[str]:
    if depth > 6:
        return []
    if isinstance(value, dict):
        lines: list[str] = []
        for key, nested in value.items():
            child = f"{path}.{key}" if path else str(key)
            if NOISE_PATH.search(child) is not None:
                continue
            lines.extend(profile_lines(nested, child, depth + 1))
        return lines
    if isinstance(value, list):
        lines = []
        for index, item in enumerate(value[:LIST_LIMIT]):
            lines.extend(profile_lines(item, f"{path}[{index}]", depth + 1))
        return lines
    if isinstance(value, bool):
        return []
    text = " ".join(str(value).split())
    if text == "" or len(text) > VALUE_LIMIT or text.startswith(("http://", "https://", "data:")):
        return []
    return [f"{path}: {text}"]


def compact_profile(payload: Any) -> str:
    lines: list[str] = []
    seen: set[str] = set()
    used = 0
    for line in profile_lines(redact_profile(payload)):
        if line in seen:
            continue
        if used + len(line) + 1 > PROFILE_LIMIT:
            break
        seen.add(line)
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)


def profile_memory(payload: Any) -> ProfileMemory:
    profile = compact_profile(payload)
    if profile == "":
        raise ProfileExtractionFailed("The profile response was empty.")
    try:
        parsed = generate_json(
            PROFILE_PROMPT.format(profile=profile, limit=FACT_LIMIT),
            PROFILE_SCHEMA,
            180,
            load_settings().planner_model,
            OUTPUT_TOKENS,
        )
    except (ChatModelFailed, TypeError, ValueError) as error:
        raise ProfileExtractionFailed("The model could not read the profile.") from error
    display_name = clean_fact(str(parsed.get("displayName", "")))
    if SECRET.search(display_name) is not None or "@" in display_name:
        display_name = ""
    raw = parsed.get("facts", [])
    facts: list[str] = []
    seen: set[str] = set()
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        label = clean_fact(str(item.get("label", "")))
        value = clean_fact(str(item.get("value", "")))
        if label == "" or value == "" or PLACEHOLDER.fullmatch(value) is not None:
            continue
        fact = f"{label}: {value}"
        if SECRET.search(fact) is not None or fact.lower() in seen:
            continue
        seen.add(fact.lower())
        facts.append(fact)
    if display_name == "" and facts == []:
        raise ProfileExtractionFailed("The profile did not include details worth remembering.")
    return ProfileMemory(display_name=display_name, facts=facts[:FACT_LIMIT])
