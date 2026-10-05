import uuid
from datetime import datetime, timezone

from app.core.database import open_app_database

VISIBLE_TEXT_LIMIT = 4000


def save_snapshot(client_id: str, site_host: str, page_url: str, visible_text: str) -> None:
    captured_at = datetime.now(timezone.utc).isoformat()
    stored_text = visible_text.strip()[:VISIBLE_TEXT_LIMIT]
    with open_app_database() as connection:
        connection.execute(
            """
            DELETE FROM portal_snapshots
            WHERE client_id = ? AND site_host = ?
            """,
            (client_id, site_host),
        )
        connection.execute(
            """
            INSERT INTO portal_snapshots (id, client_id, site_host, page_url, visible_text, captured_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (str(uuid.uuid4()), client_id, site_host, page_url.strip(), stored_text, captured_at),
        )


def read_snapshot(client_id: str, site_host: str) -> str | None:
    with open_app_database() as connection:
        row = connection.execute(
            """
            SELECT visible_text
            FROM portal_snapshots
            WHERE client_id = ? AND site_host = ?
            ORDER BY captured_at DESC
            LIMIT 1
            """,
            (client_id, site_host),
        ).fetchone()
    if row is None:
        return None
    return str(row["visible_text"])
