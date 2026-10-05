from datetime import datetime, timezone

from fastapi import HTTPException

from app.core.database import open_app_database
from app.organizations.service import normalize_site_host


def save_answer_feedback(
    message_id: str,
    session_id: str,
    client_id: str,
    site_host: str,
    rating: str,
    note: str,
) -> tuple[str, str]:
    cleaned_rating = rating.strip().lower()
    if cleaned_rating not in {"up", "down"}:
        raise HTTPException(status_code=400, detail="Choose helpful or not helpful.")
    cleaned_note = note.strip()
    if len(cleaned_note) > 300:
        raise HTTPException(status_code=400, detail="Keep the note to 300 characters or fewer.")
    if cleaned_rating == "up":
        cleaned_note = ""

    host = normalize_site_host(site_host)
    with open_app_database() as connection:
        message = connection.execute(
            """
            SELECT message.run_id
            FROM chat_messages message
            JOIN chat_sessions session ON session.id = message.session_id
            WHERE message.id = ?
              AND message.session_id = ?
              AND message.role = 'assistant'
              AND session.client_id = ?
              AND session.site_host = ?
            """,
            (message_id.strip(), session_id.strip(), client_id.strip(), host),
        ).fetchone()
        if message is None or str(message["run_id"]) == "":
            raise HTTPException(status_code=404, detail="That answer was not found.")
        now = datetime.now(timezone.utc).isoformat()
        connection.execute(
            """
            INSERT INTO answer_feedback (message_id, run_id, rating, note, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(message_id) DO UPDATE SET
                rating = excluded.rating,
                note = excluded.note,
                updated_at = excluded.updated_at
            """,
            (message_id.strip(), str(message["run_id"]), cleaned_rating, cleaned_note, now, now),
        )
    return cleaned_rating, cleaned_note
