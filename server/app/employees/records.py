import re
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone

from app.core.database import open_app_database
from app.memory.service import RECORD_SOURCE, clean_fact, forget_conversation_facts, parse_joining_date, save_fact
from app.organizations.service import find_organization_by_host

EMAIL = re.compile(r"^[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}$", re.IGNORECASE)
NAME_PREFIX = re.compile(r"(?i)^(my name is|name is|i am|i'm)\s+")
QUESTION = re.compile(r"\?|\b(how many|what is|what's|can i|could i|leave policy)\b", re.IGNORECASE)
CODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,31}")
ISO_DATE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
SLASH_DATE = re.compile(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})")
MONTH_FIRST = re.compile(r"([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})")

STEPS: tuple[tuple[str, str], ...] = (
    ("display_name", "Full name"),
    ("role_title", "Role"),
    ("joining_date", "Joining date"),
    ("work_location", "Work location"),
)


@dataclass(frozen=True)
class EmployeeRecord:
    id: str
    organization_id: str
    email: str
    display_name: str
    employee_code: str
    department: str
    role_title: str
    joining_date: str
    work_location: str
    onboarded_at: str
    updated_at: str

    def value(self, field: str) -> str:
        return str(getattr(self, field))


def normalize_email(value: str) -> str:
    cleaned = value.strip().lower()
    if EMAIL.fullmatch(cleaned) is None:
        return ""
    return cleaned


def employee_scope(employee_id: str) -> str:
    return f"employee:{employee_id}"


def _row(row: sqlite3.Row) -> EmployeeRecord:
    return EmployeeRecord(
        id=str(row["id"]),
        organization_id=str(row["organization_id"]),
        email=str(row["email"]),
        display_name=str(row["display_name"]),
        employee_code=str(row["employee_code"]),
        department=str(row["department"]),
        role_title=str(row["role_title"]),
        joining_date=str(row["joining_date"]),
        work_location=str(row["work_location"]),
        onboarded_at=str(row["onboarded_at"]),
        updated_at=str(row["updated_at"]),
    )


def read_employee(organization_id: str, email: str) -> EmployeeRecord | None:
    cleaned = normalize_email(email)
    if organization_id == "" or cleaned == "":
        return None
    with open_app_database() as connection:
        row = connection.execute(
            """
            SELECT id, organization_id, email, display_name, employee_code, department,
                   role_title, joining_date, work_location, onboarded_at, updated_at
            FROM employees
            WHERE organization_id = ? AND email = ?
            """,
            (organization_id, cleaned),
        ).fetchone()
    if row is None:
        return None
    return _row(row)


def ensure_employee(organization_id: str, email: str) -> EmployeeRecord:
    cleaned = normalize_email(email)
    existing = read_employee(organization_id, cleaned)
    if existing is not None:
        return existing
    now = datetime.now(timezone.utc).isoformat()
    employee_id = str(uuid.uuid4())
    with open_app_database() as connection:
        connection.execute(
            """
            INSERT INTO employees (
                id, organization_id, email, display_name, employee_code, department,
                role_title, joining_date, work_location, onboarded_at, created_at, updated_at
            )
            VALUES (?, ?, ?, '', '', '', '', '', '', '', ?, ?)
            """,
            (employee_id, organization_id, cleaned, now, now),
        )
    created = read_employee(organization_id, cleaned)
    if created is None:
        raise RuntimeError("The employee record was not saved.")
    return created


def list_employees(organization_id: str) -> list[EmployeeRecord]:
    with open_app_database() as connection:
        rows = connection.execute(
            """
            SELECT id, organization_id, email, display_name, employee_code, department,
                   role_title, joining_date, work_location, onboarded_at, updated_at
            FROM employees
            WHERE organization_id = ?
            ORDER BY updated_at DESC, email
            """,
            (organization_id,),
        ).fetchall()
    return [_row(row) for row in rows]


def scoped_client_id(browser_client_id: str, site_host: str, employee_email: str) -> str:
    """Use the employee record when the page address names an email, so one browser can switch people."""
    email = normalize_email(employee_email)
    if email == "":
        return browser_client_id.strip()
    organization = find_organization_by_host(site_host)
    return employee_scope(ensure_employee(organization.id, email).id)


def next_field(employee: EmployeeRecord) -> str:
    for field, _question in STEPS:
        if employee.value(field).strip() == "":
            return field
    return ""


def field_question(field: str) -> str:
    for name, question in STEPS:
        if name == field:
            return question
    return ""


def opening_question(employee: EmployeeRecord) -> str:
    field = next_field(employee)
    if field == "":
        return ""
    if field == "display_name":
        return f"I'll keep a private record for {employee.email}. {field_question(field)}"
    return field_question(field)


def _canonical_date(text: str) -> str:
    cleaned = clean_fact(text)
    iso = ISO_DATE.fullmatch(cleaned)
    slash = SLASH_DATE.fullmatch(cleaned)
    month_first = MONTH_FIRST.fullmatch(cleaned)
    try:
        if iso is not None:
            parsed = date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
        elif slash is not None:
            parsed = date(int(slash.group(3)), int(slash.group(2)), int(slash.group(1)))
        elif month_first is not None:
            parsed = datetime.strptime(f"{month_first.group(1)} {month_first.group(2)} {month_first.group(3)}", "%B %d %Y").date()
        else:
            parsed = parse_joining_date(f"I joined on {cleaned}")
    except ValueError:
        return ""
    if parsed is None:
        return ""
    return f"{parsed.day} {parsed.strftime('%B %Y')}"


def _short_text(text: str) -> str:
    value = clean_fact(text).strip(" .")
    if len(value) < 2 or len(value) > 80 or "?" in value or re.search(r"[A-Za-z]", value) is None:
        return ""
    return value


def parse_field(field: str, message: str) -> str:
    if field == "display_name":
        value = clean_fact(NAME_PREFIX.sub("", clean_fact(message))).strip(" .")
        if "@" in value:
            return ""
        return _short_text(value)
    if field == "employee_code":
        value = clean_fact(message).replace(" ", "")
        if CODE.fullmatch(value) is None:
            return ""
        return value
    if field == "joining_date":
        return _canonical_date(message)
    if field in {"department", "role_title", "work_location"}:
        return _short_text(message)
    return ""


def _save_column(employee: EmployeeRecord, field: str, value: str) -> EmployeeRecord:
    if field not in {name for name, _question in STEPS}:
        raise ValueError("Unknown employee field.")
    now = datetime.now(timezone.utc).isoformat()
    with open_app_database() as connection:
        connection.execute(
            f"UPDATE employees SET {field} = ?, updated_at = ? WHERE id = ?",
            (value, now, employee.id),
        )
    updated = read_employee(employee.organization_id, employee.email)
    if updated is None:
        raise RuntimeError("The employee record was not saved.")
    return updated


def _fact_pair(field: str, value: str, email: str) -> tuple[str, str] | None:
    if field == "display_name":
        return value, "name"
    if field == "role_title":
        return value, "role"
    if field == "joining_date":
        return value, "joining_date"
    if field == "work_location":
        return value, "office"
    if field == "email":
        return email, "email"
    return None


def _remember(employee: EmployeeRecord, field: str, value: str, source_message_id: str) -> None:
    pair = _fact_pair(field, value, employee.email)
    if pair is None:
        return
    fact, kind = pair
    save_fact(employee.organization_id, employee_scope(employee.id), fact, kind, source_message_id)


def ensure_record_facts(employee: EmployeeRecord) -> None:
    scope = employee_scope(employee.id)
    pairs: list[tuple[str, str]] = []
    for field, _label in STEPS:
        value = employee.value(field).strip()
        if value == "":
            continue
        pair = _fact_pair(field, value, employee.email)
        if pair is not None:
            pairs.append(pair)
    email_pair = _fact_pair("email", employee.email, employee.email)
    if email_pair is not None:
        pairs.append(email_pair)
    for fact, kind in pairs:
        save_fact(employee.organization_id, scope, fact, kind, RECORD_SOURCE)


def delete_employee(organization_id: str, email: str) -> bool:
    """Remove the saved employee record, its facts, and its chat. Memory notes go with the record."""
    employee = read_employee(organization_id, email)
    if employee is None:
        return False
    scope = employee_scope(employee.id)
    with open_app_database() as connection:
        sessions = connection.execute(
            "SELECT id FROM chat_sessions WHERE organization_id = ? AND client_id = ?",
            (organization_id, scope),
        ).fetchall()
        session_ids = [str(row["id"]) for row in sessions]
        if session_ids != []:
            marks = ",".join("?" for _ in session_ids)
            connection.execute(
                f"""
                DELETE FROM answer_feedback
                WHERE message_id IN (
                    SELECT id FROM chat_messages WHERE session_id IN ({marks})
                )
                """,
                session_ids,
            )
            connection.execute(
                f"DELETE FROM chat_messages WHERE session_id IN ({marks})",
                session_ids,
            )
            connection.execute(
                f"DELETE FROM chat_sessions WHERE id IN ({marks})",
                session_ids,
            )
        connection.execute(
            "DELETE FROM employee_facts WHERE organization_id = ? AND client_id = ?",
            (organization_id, scope),
        )
        connection.execute("DELETE FROM employees WHERE id = ?", (employee.id,))
    return True


def save_onboarding(
    organization_id: str,
    email: str,
    display_name: str,
    role_title: str,
    joining_date: str,
    work_location: str,
) -> EmployeeRecord:
    """Save the onboarding form as plain values and mark the employee ready to chat."""
    employee = ensure_employee(organization_id, email)
    name = parse_field("display_name", display_name)
    role = parse_field("role_title", role_title)
    joined = parse_field("joining_date", joining_date)
    location = parse_field("work_location", work_location)
    if name == "":
        raise ValueError("Enter your full name.")
    if role == "":
        raise ValueError("Enter your role.")
    if joined == "":
        raise ValueError("Enter a joining date such as 12 March 2024.")
    if location == "":
        raise ValueError("Enter your work location.")
    now = datetime.now(timezone.utc).isoformat()
    with open_app_database() as connection:
        connection.execute(
            """
            UPDATE employees
            SET display_name = ?, role_title = ?, joining_date = ?, work_location = ?,
                onboarded_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (name, role, joined, location, now, now, employee.id),
        )
        connection.execute(
            """
            DELETE FROM employee_facts
            WHERE organization_id = ? AND client_id = ? AND kind IN ('employee_code', 'department')
            """,
            (organization_id, employee_scope(employee.id)),
        )
    saved = read_employee(organization_id, employee.email)
    if saved is None:
        raise RuntimeError("The employee record was not saved.")
    ensure_record_facts(saved)
    return saved


def onboarding_reply(employee: EmployeeRecord, message: str, source_message_id: str) -> tuple[str, str]:
    """Store the answer for the next empty field. Returns the reply and its answer kind."""
    field = next_field(employee)
    if field == "":
        return (
            f"I already have your record for {employee.email}. You can ask me about your organization's leave policy.",
            "onboarding_done",
        )
    question = field_question(field)
    if QUESTION.search(message) is not None:
        return (f"I'll answer that after I have your details. {question}", "onboarding")
    value = parse_field(field, message)
    if value == "":
        return (f"I still need that. {question}", "onboarding")
    updated = _save_column(employee, field, value)
    _remember(updated, field, value, source_message_id)
    following = next_field(updated)
    if following != "":
        return (field_question(following), "onboarding")
    now = datetime.now(timezone.utc).isoformat()
    with open_app_database() as connection:
        connection.execute(
            "UPDATE employees SET onboarded_at = ?, updated_at = ? WHERE id = ? AND onboarded_at = ''",
            (now, now, updated.id),
        )
    finished = read_employee(updated.organization_id, updated.email)
    if finished is not None:
        ensure_record_facts(finished)
        name = finished.display_name
    else:
        name = value
    return (
        f"Thanks, {name}. I've saved your details for {finished.email if finished is not None else employee.email}. "
        "You can ask me about your organization's leave policy.",
        "onboarding_done",
    )


def forget_notes(organization_id: str, scope: str) -> int:
    """Remove chat notes and keep the onboarding record."""
    return forget_conversation_facts(organization_id, scope)
