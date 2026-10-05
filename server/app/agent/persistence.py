import sqlite3
from pathlib import Path

from crewai.flow.persistence import SQLiteFlowPersistence

from app.core.settings import load_settings

_store: SQLiteFlowPersistence | None = None


def flow_database_path() -> Path:
    settings = load_settings()
    path = Path(settings.flow_database)
    if not path.is_absolute():
        path = Path.cwd() / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def flow_store() -> SQLiteFlowPersistence:
    """One SQLite store for every chat session, at data/flows.db."""
    global _store
    if _store is None:
        _store = SQLiteFlowPersistence(str(flow_database_path()))
    return _store


def delete_flow_sessions(session_ids: list[str]) -> None:
    """Remove persisted flow state for these sessions. Other organizations stay."""
    if session_ids == []:
        return
    path = flow_database_path()
    if not path.is_file():
        return
    flow_store()
    marks = ",".join("?" for _ in session_ids)
    with sqlite3.connect(path, timeout=30) as connection:
        connection.execute(f"DELETE FROM flow_states WHERE flow_uuid IN ({marks})", session_ids)
        connection.execute(
            f"DELETE FROM pending_feedback WHERE flow_uuid IN ({marks})",
            session_ids,
        )
