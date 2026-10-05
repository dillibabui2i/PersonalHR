import re
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone

from app.core.database import open_app_database

SECRET = re.compile(
    r"\b(password|passcode|pin|otp|one[- ]time password|api key|access token|secret|credential)\b",
    re.IGNORECASE,
)
REMEMBER_PREFIX = re.compile(r"^\s*(?:please\s+)?remember(?:\s+that)?\s+", re.IGNORECASE)
REMEMBER_SUFFIX = re.compile(
    r"(?:[,.;]\s*|\s+)(?:please\s+)?remember(?:\s+(?:it|this|that))?[.!]?\s*$",
    re.IGNORECASE,
)
JOINED = re.compile(r"\bI\s+(?:joined|started(?:\s+work)?)\b", re.IGNORECASE)
OFFICE = re.compile(r"\bI\s+(?:work|am based)\s+(?:from|in|at)\b", re.IGNORECASE)
DATE_TEXT = re.compile(r"\b(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})\b")
QUESTION_LIKE = re.compile(
    r"\?|\b(how many|how much|when can|when do|when will|when i|can i|could i|what is|what's|tell me)\b",
    re.IGNORECASE,
)
LEAVE_BALANCE = re.compile(
    r"\b(?:i have|i've got|i currently have|my balance is)\b.{0,40}\b\d+(?:\.\d+)?\b",
    re.IGNORECASE,
)
POLICY_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\s+days?\b", re.IGNORECASE)


@dataclass(frozen=True)
class EmployeeFact:
    id: str
    fact: str
    kind: str
    created_at: str
    source: str = ""


def clean_fact(value: str) -> str:
    cleaned = " ".join(value.strip().split())
    return cleaned.rstrip(" .")


def durable_fact(fact: str) -> bool:
    """True for a short personal detail, not a policy question or leave balance."""
    if fact == "" or len(fact) > 140 or QUESTION_LIKE.search(fact) is not None:
        return False
    if LEAVE_BALANCE.search(fact) is not None:
        return False
    return POLICY_NUMBER.search(fact) is None or JOINED.search(fact) is not None


def fact_kind(fact: str) -> str:
    lowered = fact.lower()
    if JOINED.search(fact) is not None or "joining date" in lowered:
        return "joining_date"
    if "@" in fact and re.search(r"\bemail\b", lowered) is not None:
        return "email"
    if OFFICE.search(fact) is not None or "office" in lowered or "location" in lowered:
        return "office"
    if re.search(r"\b(role|title|designation|position)\b", lowered) is not None:
        return "role"
    if "employee code" in lowered or "employee id" in lowered:
        return "employee_code"
    if "department" in lowered:
        return "department"
    if re.search(r"\bteam\b", lowered) is not None:
        return "team"
    if "manager" in lowered:
        return "manager"
    if re.search(r"\bname is\b", lowered) is not None:
        return "name"
    return "general"


def sanitize_stated_facts(facts: list[str]) -> list[str]:
    """Keep only non-empty, non-secret facts with stable wording."""
    cleaned: list[str] = []
    seen: set[str] = set()
    for fact in facts:
        value = clean_fact(fact)
        value = clean_fact(REMEMBER_SUFFIX.sub("", REMEMBER_PREFIX.sub("", value)))
        value = clean_fact(
            re.sub(r"\b(?:please\s+)?remember(?:\s+(?:it|this|that))?\b", "", value, flags=re.IGNORECASE)
        )
        if value == "" or SECRET.search(value) is not None or not durable_fact(value):
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(value)
    return cleaned


PERSONAL_CLAUSE = re.compile(
    r"\b(?:"
    r"my name is\s+[^.,!?]{2,40}"
    r"|I\s+(?:joined|started(?:\s+work)?)\s+[^.,!?]{2,60}"
    r"|I\s+(?:work|am based)\s+(?:from|in|at)\s+[^.,!?]{2,40}"
    r"|I\s+work as\s+[^.,!?]{2,40}"
    r"|my (?:manager|role|title|designation|position|team|department|office) is\s+[^.,!?]{2,40}"
    r")",
    re.IGNORECASE,
)


def heuristic_stated_facts(message: str) -> list[str]:
    """Facts the employee stated about themself, including inside a longer question."""
    cleaned = clean_fact(message)
    if cleaned == "" or SECRET.search(cleaned) is not None:
        return []
    explicit = clean_fact(REMEMBER_SUFFIX.sub("", REMEMBER_PREFIX.sub("", cleaned)))
    if re.search(r"\bremember\b", explicit, re.IGNORECASE) is not None:
        explicit = clean_fact(re.sub(r"\b(?:please\s+)?remember(?:\s+(?:it|this|that))?\b", "", explicit, flags=re.IGNORECASE))
    found: list[str] = []
    if explicit != "" and explicit != cleaned and not re.fullmatch(r"remember", explicit, re.IGNORECASE):
        found.append(explicit)
    elif re.search(r"\bremember\b", cleaned, re.IGNORECASE) is not None and explicit not in {"", "remember"}:
        found.append(explicit)
    for match in PERSONAL_CLAUSE.finditer(cleaned):
        clause = clean_fact(match.group(0))
        if clause != "" and clause.lower() not in {item.lower() for item in found}:
            found.append(clause)
    return found


def stated_facts(message: str, planned: list[str] | None = None) -> list[str]:
    """Keep planner facts and any personal fact stated in the message."""
    found = sanitize_stated_facts([] if planned is None else planned)
    seen = {fact.lower() for fact in found}
    for fact in sanitize_stated_facts(heuristic_stated_facts(message)):
        if fact.lower() in seen:
            continue
        seen.add(fact.lower())
        found.append(fact)
    return found


def refuses_memory(message: str, facts: list[str]) -> bool:
    """True when the employee asked to keep something secret or unsafe."""
    cleaned = clean_fact(message)
    if cleaned == "" or facts != [] or SECRET.search(cleaned) is None:
        return False
    return (
        re.search(r"\bremember\b", cleaned, re.IGNORECASE) is not None
        or re.search(r"\b(keep|save|store)\b", cleaned, re.IGNORECASE) is not None
    )


def is_page_figure(fact: str, page_text: str | None) -> bool:
    if page_text is None or page_text.strip() == "":
        return False
    numbers = re.findall(r"\d+(?:\.\d+)?", fact)
    return numbers != [] and all(re.search(rf"(?<!\d){re.escape(number)}(?!\d)", page_text) for number in numbers)


REPLACEABLE_KINDS = frozenset(
    {"joining_date", "office", "name", "email", "role", "department", "team", "manager", "employee_code"}
)
PROFILE_SOURCE = "portal-profile"
RECORD_SOURCE = "employee-record"
RECORD_SOURCES = frozenset({PROFILE_SOURCE, RECORD_SOURCE})
LEGACY_PROFILE_KINDS = ("name", "email", "role")


def employee_display_name(facts: list[EmployeeFact]) -> str:
    for fact in facts:
        if fact.kind != "name":
            continue
        value = clean_fact(fact.fact)
        match = re.match(r"(?i)^(?:my name is|name is|name:|i am|i'm)\s*(.+)$", value)
        if match is not None:
            return clean_fact(match.group(1))
        return value
    return ""


def save_fact(
    organization_id: str,
    client_id: str,
    fact: str,
    kind: str,
    source_message_id: str,
    page_text: str | None = None,
) -> bool:
    cleaned = clean_fact(fact)
    if (
        organization_id == ""
        or client_id.strip() == ""
        or cleaned == ""
        or SECRET.search(cleaned) is not None
        or is_page_figure(cleaned, page_text)
        or (source_message_id not in RECORD_SOURCES and not durable_fact(cleaned))
    ):
        return False
    now = datetime.now(timezone.utc).isoformat()
    with open_app_database() as connection:
        existing = connection.execute(
            """
            SELECT id FROM employee_facts
            WHERE organization_id = ? AND client_id = ? AND lower(fact) = lower(?)
            """,
            (organization_id, client_id, cleaned),
        ).fetchone()
        if existing is not None:
            return False
        if kind in REPLACEABLE_KINDS:
            prior = connection.execute(
                """
                SELECT id FROM employee_facts
                WHERE organization_id = ? AND client_id = ? AND kind = ?
                ORDER BY created_at DESC, rowid DESC
                LIMIT 1
                """,
                (organization_id, client_id, kind),
            ).fetchone()
            if prior is not None:
                connection.execute(
                    """
                    UPDATE employee_facts
                    SET fact = ?, source_message_id = ?, created_at = ?, last_used_at = ''
                    WHERE id = ?
                    """,
                    (cleaned, source_message_id, now, str(prior["id"])),
                )
                return True
        connection.execute(
            """
            INSERT INTO employee_facts (
                id, organization_id, client_id, fact, kind, source_message_id, created_at, last_used_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, '')
            """,
            (str(uuid.uuid4()), organization_id, client_id, cleaned, kind, source_message_id, now),
        )
    return True


def profile_loaded(organization_id: str, client_id: str) -> bool:
    if organization_id == "" or client_id.strip() == "":
        return False
    with open_app_database() as connection:
        row = connection.execute(
            """
            SELECT 1 FROM employee_facts
            WHERE organization_id = ? AND client_id = ? AND source_message_id = ?
            LIMIT 1
            """,
            (organization_id, client_id.strip(), PROFILE_SOURCE),
        ).fetchone()
    return row is not None


def replace_profile_facts(organization_id: str, client_id: str, display_name: str, facts: list[str]) -> int:
    """Swap the portal-profile snapshot for this employee; conversation facts stay untouched."""
    cleaned_client = client_id.strip()
    if organization_id == "" or cleaned_client == "":
        return 0
    rows: list[tuple[str, str]] = []
    if display_name != "":
        rows.append((display_name, "name"))
    rows.extend((fact, "profile") for fact in facts)
    now = datetime.now(timezone.utc).isoformat()
    marks = ",".join("?" for _ in LEGACY_PROFILE_KINDS)
    with open_app_database() as connection:
        connection.execute(
            f"""
            DELETE FROM employee_facts
            WHERE organization_id = ? AND client_id = ?
              AND (source_message_id = ? OR (source_message_id = '' AND kind IN ({marks})))
            """,
            (organization_id, cleaned_client, PROFILE_SOURCE, *LEGACY_PROFILE_KINDS),
        )
        for fact, kind in rows:
            connection.execute(
                """
                INSERT INTO employee_facts (
                    id, organization_id, client_id, fact, kind, source_message_id, created_at, last_used_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, '')
                """,
                (str(uuid.uuid4()), organization_id, cleaned_client, fact, kind, PROFILE_SOURCE, now),
            )
    return len(rows)


def recall_facts(
    organization_id: str,
    client_id: str,
    limit: int,
    conversation_only: bool = False,
) -> list[EmployeeFact]:
    if organization_id == "" or client_id.strip() == "" or limit <= 0:
        return []
    source_filter = ""
    params: list[object] = [organization_id, client_id]
    if conversation_only:
        marks = ",".join("?" for _ in RECORD_SOURCES)
        source_filter = f"AND source_message_id NOT IN ({marks})"
        params.extend(sorted(RECORD_SOURCES))
    params.append(limit)
    with open_app_database() as connection:
        rows = connection.execute(
            f"""
            SELECT id, fact, kind, created_at, source_message_id
            FROM employee_facts
            WHERE organization_id = ? AND client_id = ? {source_filter}
            ORDER BY created_at DESC, rowid DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
    return [
        EmployeeFact(
            id=str(row["id"]),
            fact=str(row["fact"]),
            kind=str(row["kind"]),
            created_at=str(row["created_at"]),
            source=str(row["source_message_id"]),
        )
        for row in rows
    ]


def forget_fact(organization_id: str, client_id: str, fact_id: str) -> bool:
    if organization_id == "" or client_id.strip() == "" or fact_id.strip() == "":
        return False
    marks = ",".join("?" for _ in RECORD_SOURCES)
    with open_app_database() as connection:
        cursor = connection.execute(
            f"""
            DELETE FROM employee_facts
            WHERE id = ? AND organization_id = ? AND client_id = ?
              AND source_message_id NOT IN ({marks})
            """,
            (fact_id, organization_id, client_id.strip(), *sorted(RECORD_SOURCES)),
        )
    return cursor.rowcount > 0


def forget_all_facts(organization_id: str, client_id: str) -> int:
    if organization_id == "" or client_id.strip() == "":
        return 0
    with open_app_database() as connection:
        cursor = connection.execute(
            """
            DELETE FROM employee_facts
            WHERE organization_id = ? AND client_id = ?
            """,
            (organization_id, client_id.strip()),
        )
    return cursor.rowcount


def forget_conversation_facts(organization_id: str, client_id: str) -> int:
    """Remove chat notes and keep onboarding and portal-profile snapshots."""
    if organization_id == "" or client_id.strip() == "":
        return 0
    marks = ",".join("?" for _ in RECORD_SOURCES)
    with open_app_database() as connection:
        cursor = connection.execute(
            f"""
            DELETE FROM employee_facts
            WHERE organization_id = ? AND client_id = ? AND source_message_id NOT IN ({marks})
            """,
            (organization_id, client_id.strip(), *sorted(RECORD_SOURCES)),
        )
    return cursor.rowcount


def parse_joining_date(fact: str) -> date | None:
    match = DATE_TEXT.search(fact)
    if match is None:
        return None
    try:
        return datetime.strptime(" ".join(match.groups()), "%d %B %Y").date()
    except ValueError:
        try:
            return datetime.strptime(" ".join(match.groups()), "%d %b %Y").date()
        except ValueError:
            return None


def tenure_text(joined: date, today: date | None = None) -> str:
    current = date.today() if today is None else today
    months = (current.year - joined.year) * 12 + current.month - joined.month
    if current.day < joined.day:
        months -= 1
    years, remaining = divmod(max(months, 0), 12)
    return f"{years} years and {remaining} months as of {current.strftime('%d %B %Y')}"


FACT_LABELS = {
    "name": "Name",
    "email": "Work email",
    "role": "Role",
    "joining_date": "Joining date",
    "office": "Work location",
    "department": "Department",
    "employee_code": "Employee code",
    "manager": "Manager",
    "team": "Team",
}


def facts_context(facts: list[EmployeeFact]) -> str:
    lines: list[str] = []
    for fact in facts:
        label = FACT_LABELS.get(fact.kind, "Fact")
        value = employee_display_name([fact]) if fact.kind == "name" else fact.fact
        line = f"- {label}: {value}"
        if fact.kind == "joining_date":
            joined = parse_joining_date(fact.fact)
            if joined is not None:
                line += f" (calculated tenure: {tenure_text(joined)})"
        lines.append(line)
    return "\n".join(lines)


def mark_facts_used(facts: list[EmployeeFact]) -> None:
    if facts == []:
        return
    now = datetime.now(timezone.utc).isoformat()
    ids = [fact.id for fact in facts]
    marks = ",".join("?" for _ in ids)
    with open_app_database() as connection:
        connection.execute(
            f"UPDATE employee_facts SET last_used_at = ? WHERE id IN ({marks})",
            [now, *ids],
        )
