import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from app.core.database import open_app_database
from app.rag.types import Citation


def create_session(client_id: str, site_host: str, organization_id: str) -> str:
    session_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    with open_app_database() as connection:
        connection.execute(
            """
            INSERT INTO chat_sessions (id, client_id, site_host, organization_id, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, client_id, site_host, organization_id, created_at),
        )
    return session_id


def session_matches(session_id: str, client_id: str, site_host: str) -> bool:
    with open_app_database() as connection:
        row = connection.execute(
            """
            SELECT id
            FROM chat_sessions
            WHERE id = ? AND client_id = ? AND site_host = ?
            """,
            (session_id, client_id, site_host),
        ).fetchone()
    return row is not None


def session_is_open(session_id: str, client_id: str, site_host: str) -> bool:
    with open_app_database() as connection:
        row = connection.execute(
            """
            SELECT id
            FROM chat_sessions
            WHERE id = ? AND client_id = ? AND site_host = ? AND ended_at = ''
            """,
            (session_id, client_id, site_host),
        ).fetchone()
    return row is not None


def end_session(session_id: str) -> None:
    ended_at = datetime.now(timezone.utc).isoformat()
    with open_app_database() as connection:
        connection.execute(
            "UPDATE chat_sessions SET ended_at = ? WHERE id = ? AND ended_at = ''",
            (ended_at, session_id),
        )


def start_new_conversation(session_id: str | None, client_id: str, site_host: str, organization_id: str) -> str:
    """End the current session and open an empty one. Stored facts are left in place."""
    current = (session_id or "").strip()
    if current != "" and session_matches(current, client_id, site_host):
        end_session(current)
    return create_session(client_id, site_host, organization_id)


@dataclass(frozen=True)
class StoredMessage:
    id: str
    role: str
    content: str
    citations: list[Citation]
    answer_kind: str
    created_at: str
    choices: tuple[str, ...] = ()
    suggestions: tuple[str, ...] = ()
    remembered: bool = False
    from_page: bool = False
    rating: str = ""
    feedback_note: str = ""


@dataclass(frozen=True)
class SavedMessage:
    id: str
    created_at: str


def stored_choices(raw: str) -> tuple[str, ...]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return ()
    if not isinstance(parsed, list):
        return ()
    return tuple(str(item) for item in parsed if isinstance(item, str) and str(item).strip() != "")


def stored_citations(raw: str) -> list[Citation]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    citations: list[Citation] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        document_name = item.get("documentName", "")
        section_title = item.get("sectionTitle", "")
        if not isinstance(document_name, str) or not isinstance(section_title, str):
            continue
        excerpt = item.get("excerpt", "")
        citations.append(
            Citation(
                document_name=document_name,
                section_title=section_title,
                excerpt=excerpt if isinstance(excerpt, str) else "",
            )
        )
    return citations


def list_messages(session_id: str) -> list[StoredMessage]:
    with open_app_database() as connection:
        rows = connection.execute(
            """
            SELECT message.id, message.role, message.content, message.citations_json,
                   message.answer_kind, message.choices_json, message.suggestions_json,
                   message.created_at, message.from_page,
                   COALESCE(feedback.rating, '') AS rating,
                   COALESCE(feedback.note, '') AS feedback_note,
                   EXISTS (
                       SELECT 1 FROM employee_facts fact
                       WHERE fact.source_message_id = message.id
                   ) AS remembered
            FROM chat_messages message
            LEFT JOIN answer_feedback feedback ON feedback.message_id = message.id
            WHERE session_id = ?
            ORDER BY message.created_at, message.rowid
            """,
            (session_id,),
        ).fetchall()
    return [
        StoredMessage(
            id=str(row["id"]),
            role=str(row["role"]),
            content=str(row["content"]),
            citations=stored_citations(str(row["citations_json"])),
            answer_kind=str(row["answer_kind"]),
            created_at=str(row["created_at"]),
            choices=stored_choices(str(row["choices_json"])),
            suggestions=stored_choices(str(row["suggestions_json"])),
            remembered=bool(row["remembered"]),
            from_page=bool(row["from_page"]),
            rating=str(row["rating"]),
            feedback_note=str(row["feedback_note"]),
        )
        for row in rows
    ]


def recent_turns(session_id: str, limit: int) -> list[tuple[str, str]]:
    """Earlier turns in this session, excluding the user message just saved."""
    messages = list_messages(session_id)
    if messages != [] and messages[-1].role == "user":
        messages = messages[:-1]
    user_indexes = [index for index, message in enumerate(messages) if message.role == "user"]
    if len(user_indexes) > limit:
        messages = messages[user_indexes[-limit] :]
    return [(message.role, message.content) for message in messages]


def last_user_question(session_id: str) -> str:
    with open_app_database() as connection:
        row = connection.execute(
            """
            SELECT content
            FROM chat_messages
            WHERE session_id = ? AND role = 'user'
            ORDER BY created_at DESC, rowid DESC
            LIMIT 1
            """,
            (session_id,),
        ).fetchone()
    return "" if row is None else str(row["content"])


def decline_last_no_answer(session_id: str) -> bool:
    """Keep the no-answer sentence and stop offering a general-knowledge reply."""
    with open_app_database() as connection:
        row = connection.execute(
            """
            SELECT id, answer_kind
            FROM chat_messages
            WHERE session_id = ? AND role = 'assistant'
            ORDER BY created_at DESC, rowid DESC
            LIMIT 1
            """,
            (session_id,),
        ).fetchone()
        if row is None:
            return False
        kind = str(row["answer_kind"])
        if kind == "declined":
            return True
        if kind != "no_answer":
            return False
        connection.execute(
            "UPDATE chat_messages SET answer_kind = 'declined' WHERE id = ?",
            (str(row["id"]),),
        )
    return True


def replace_last_no_answer(session_id: str, content: str, answer_kind: str, run_id: str = "") -> SavedMessage | None:
    with open_app_database() as connection:
        row = connection.execute(
            """
            SELECT id, answer_kind, created_at
            FROM chat_messages
            WHERE session_id = ? AND role = 'assistant'
            ORDER BY created_at DESC, rowid DESC
            LIMIT 1
            """,
            (session_id,),
        ).fetchone()
        if row is None or str(row["answer_kind"]) != "no_answer":
            return None
        connection.execute(
            "UPDATE chat_messages SET content = ?, citations_json = '[]', answer_kind = ?, run_id = ? WHERE id = ?",
            (content, answer_kind, run_id, str(row["id"])),
        )
    return SavedMessage(id=str(row["id"]), created_at=str(row["created_at"]))


def citation_payload(citation: Citation) -> dict[str, str]:
    payload = {
        "documentName": citation.document_name,
        "sectionTitle": citation.section_title,
    }
    if citation.excerpt != "":
        payload["excerpt"] = citation.excerpt
    return payload


def save_message(
    session_id: str,
    role: str,
    content: str,
    citations: list[Citation],
    answer_kind: str = "knowledge",
    choices: tuple[str, ...] = (),
    from_page: bool = False,
    suggestions: tuple[str, ...] = (),
    run_id: str = "",
) -> SavedMessage:
    message_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    payload = [citation_payload(citation) for citation in citations]
    with open_app_database() as connection:
        connection.execute(
            """
            INSERT INTO chat_messages (
                id, session_id, role, content, citations_json, created_at, answer_kind, choices_json,
                from_page, suggestions_json, run_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message_id,
                session_id,
                role,
                content,
                json.dumps(payload),
                created_at,
                answer_kind,
                json.dumps(list(choices)),
                1 if from_page else 0,
                json.dumps(list(suggestions)),
                run_id,
            ),
        )
    return SavedMessage(id=message_id, created_at=created_at)
