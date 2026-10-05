import sqlite3
from pathlib import Path

from app.core.settings import load_settings


def data_directory() -> Path:
    settings = load_settings()
    data_path = Path(settings.data_dir)
    if not data_path.is_absolute():
        data_path = Path.cwd() / data_path
    data_path.mkdir(parents=True, exist_ok=True)
    return data_path


LEGACY_NO_ANSWER = "I don't know anything related to this from the provided knowledge."


def add_ended_at_column(connection: sqlite3.Connection) -> None:
    columns = {str(row["name"]) for row in connection.execute("PRAGMA table_info(chat_sessions)")}
    if "ended_at" in columns:
        return
    connection.execute("ALTER TABLE chat_sessions ADD COLUMN ended_at TEXT NOT NULL DEFAULT ''")


def add_choices_column(connection: sqlite3.Connection) -> None:
    columns = {str(row["name"]) for row in connection.execute("PRAGMA table_info(chat_messages)")}
    if "choices_json" in columns:
        return
    connection.execute("ALTER TABLE chat_messages ADD COLUMN choices_json TEXT NOT NULL DEFAULT '[]'")


def add_guardrail_column(connection: sqlite3.Connection) -> None:
    columns = {str(row["name"]) for row in connection.execute("PRAGMA table_info(quality_results)")}
    if "guardrail" in columns:
        return
    connection.execute("ALTER TABLE quality_results ADD COLUMN guardrail TEXT NOT NULL DEFAULT ''")


def add_suggestions_column(connection: sqlite3.Connection) -> None:
    columns = {str(row["name"]) for row in connection.execute("PRAGMA table_info(chat_messages)")}
    if "suggestions_json" in columns:
        return
    connection.execute("ALTER TABLE chat_messages ADD COLUMN suggestions_json TEXT NOT NULL DEFAULT '[]'")


def add_from_page_column(connection: sqlite3.Connection) -> None:
    columns = {str(row["name"]) for row in connection.execute("PRAGMA table_info(chat_messages)")}
    if "from_page" in columns:
        return
    connection.execute("ALTER TABLE chat_messages ADD COLUMN from_page INTEGER NOT NULL DEFAULT 0")


def add_profile_api_columns(connection: sqlite3.Connection) -> None:
    columns = {str(row["name"]) for row in connection.execute("PRAGMA table_info(organization_settings)")}
    if "profile_api_path" not in columns:
        connection.execute(
            "ALTER TABLE organization_settings ADD COLUMN profile_api_path TEXT NOT NULL DEFAULT ''"
        )
    if "profile_name_path" not in columns:
        connection.execute(
            "ALTER TABLE organization_settings ADD COLUMN profile_name_path TEXT NOT NULL DEFAULT 'name'"
        )
    if "profile_email_path" not in columns:
        connection.execute(
            "ALTER TABLE organization_settings ADD COLUMN profile_email_path TEXT NOT NULL DEFAULT ''"
        )


def add_answer_kind_column(connection: sqlite3.Connection) -> None:
    columns = {str(row["name"]) for row in connection.execute("PRAGMA table_info(chat_messages)")}
    if "answer_kind" in columns:
        return
    connection.execute("ALTER TABLE chat_messages ADD COLUMN answer_kind TEXT NOT NULL DEFAULT 'knowledge'")
    connection.execute(
        "UPDATE chat_messages SET answer_kind = 'no_answer' WHERE role = 'assistant' AND content = ?",
        (LEGACY_NO_ANSWER,),
    )


def add_embedding_model_columns(connection: sqlite3.Connection) -> None:
    columns = {str(row["name"]) for row in connection.execute("PRAGMA table_info(organization_models)")}
    if "embed_model" not in columns:
        connection.execute(
            "ALTER TABLE organization_models ADD COLUMN embed_model TEXT NOT NULL DEFAULT ''"
        )
    if "embed_dimensions" not in columns:
        connection.execute(
            "ALTER TABLE organization_models ADD COLUMN embed_dimensions INTEGER NOT NULL DEFAULT 0"
        )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS model_kinds (
            name TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            dimensions INTEGER NOT NULL
        )
        """
    )


def add_run_id_column(connection: sqlite3.Connection) -> None:
    columns = {str(row["name"]) for row in connection.execute("PRAGMA table_info(chat_messages)")}
    if "run_id" in columns:
        return
    connection.execute("ALTER TABLE chat_messages ADD COLUMN run_id TEXT NOT NULL DEFAULT ''")


def open_app_database() -> sqlite3.Connection:
    connection = sqlite3.connect(data_directory() / "app.db")
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS organizations (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            site_host TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            organization_id TEXT NOT NULL,
            file_name TEXT NOT NULL,
            stored_path TEXT NOT NULL,
            status TEXT NOT NULL,
            error_message TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (organization_id) REFERENCES organizations(id)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id TEXT PRIMARY KEY,
            client_id TEXT NOT NULL,
            site_host TEXT NOT NULL,
            organization_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (organization_id) REFERENCES organizations(id)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS portal_snapshots (
            id TEXT PRIMARY KEY,
            client_id TEXT NOT NULL,
            site_host TEXT NOT NULL,
            page_url TEXT NOT NULL,
            visible_text TEXT NOT NULL,
            captured_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS portal_sessions (
            client_id TEXT NOT NULL,
            site_host TEXT NOT NULL,
            page_url TEXT NOT NULL,
            session_json TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (client_id, site_host)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_messages (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            citations_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES chat_sessions(id)
        )
        """
    )
    add_answer_kind_column(connection)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_runs (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            organization_id TEXT NOT NULL,
            client_id TEXT NOT NULL,
            engine TEXT NOT NULL,
            question TEXT NOT NULL,
            standalone_question TEXT NOT NULL,
            route TEXT NOT NULL,
            plan_json TEXT NOT NULL,
            status TEXT NOT NULL,
            total_ms INTEGER NOT NULL,
            first_token_ms INTEGER,
            created_at TEXT NOT NULL,
            FOREIGN KEY (organization_id) REFERENCES organizations(id)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS quality_checks (
            id TEXT PRIMARY KEY,
            organization_id TEXT NOT NULL,
            question TEXT NOT NULL,
            expected_phrase TEXT NOT NULL,
            expected_document TEXT NOT NULL,
            expect_no_answer INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (organization_id) REFERENCES organizations(id)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS quality_runs (
            id TEXT PRIMARY KEY,
            organization_id TEXT NOT NULL,
            status TEXT NOT NULL,
            check_ids_json TEXT NOT NULL,
            current_index INTEGER NOT NULL,
            detail TEXT NOT NULL,
            created_at TEXT NOT NULL,
            finished_at TEXT NOT NULL,
            FOREIGN KEY (organization_id) REFERENCES organizations(id)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS quality_results (
            id TEXT PRIMARY KEY,
            check_id TEXT NOT NULL,
            engine TEXT NOT NULL,
            passed INTEGER NOT NULL,
            answer TEXT NOT NULL,
            citations_json TEXT NOT NULL,
            total_ms INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (check_id) REFERENCES quality_checks(id)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_steps (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            name TEXT NOT NULL,
            input_summary TEXT NOT NULL,
            output_summary TEXT NOT NULL,
            status TEXT NOT NULL,
            started_ms INTEGER NOT NULL,
            ended_ms INTEGER NOT NULL,
            FOREIGN KEY (run_id) REFERENCES agent_runs(id)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS answer_feedback (
            message_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            rating TEXT NOT NULL,
            note TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (message_id) REFERENCES chat_messages(id),
            FOREIGN KEY (run_id) REFERENCES agent_runs(id)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS employee_facts (
            id TEXT PRIMARY KEY,
            organization_id TEXT NOT NULL,
            client_id TEXT NOT NULL,
            fact TEXT NOT NULL,
            kind TEXT NOT NULL,
            source_message_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            last_used_at TEXT NOT NULL,
            FOREIGN KEY (organization_id) REFERENCES organizations(id)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS answer_cache (
            id TEXT PRIMARY KEY,
            organization_id TEXT NOT NULL,
            document_set_key TEXT NOT NULL,
            question_key TEXT NOT NULL,
            answer TEXT NOT NULL,
            citations_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (organization_id) REFERENCES organizations(id),
            UNIQUE (organization_id, document_set_key, question_key)
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS answer_cache_scope
        ON answer_cache (organization_id, document_set_key, question_key)
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS employee_facts_scope
        ON employee_facts (organization_id, client_id, created_at)
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS organization_settings (
            organization_id TEXT PRIMARY KEY,
            greeting TEXT NOT NULL,
            answer_style TEXT NOT NULL,
            portal_enabled INTEGER NOT NULL,
            memory_enabled INTEGER NOT NULL,
            starter_questions_json TEXT NOT NULL,
            profile_api_path TEXT NOT NULL DEFAULT '',
            profile_name_path TEXT NOT NULL DEFAULT 'name',
            profile_email_path TEXT NOT NULL DEFAULT '',
            FOREIGN KEY (organization_id) REFERENCES organizations(id)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS admin_users (
            id TEXT PRIMARY KEY,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL,
            organization_id TEXT,
            active INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (organization_id) REFERENCES organizations(id)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS organization_models (
            organization_id TEXT PRIMARY KEY,
            chat_model TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (organization_id) REFERENCES organizations(id)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS model_switch_requests (
            id TEXT PRIMARY KEY,
            organization_id TEXT NOT NULL,
            requested_model TEXT NOT NULL,
            reason TEXT NOT NULL,
            status TEXT NOT NULL,
            requested_by TEXT NOT NULL,
            reviewed_by TEXT NOT NULL,
            review_note TEXT NOT NULL,
            created_at TEXT NOT NULL,
            reviewed_at TEXT NOT NULL,
            FOREIGN KEY (organization_id) REFERENCES organizations(id)
        )
        """
    )
    add_ended_at_column(connection)
    add_guardrail_column(connection)
    add_choices_column(connection)
    add_from_page_column(connection)
    add_suggestions_column(connection)
    add_run_id_column(connection)
    add_profile_api_columns(connection)
    add_embedding_model_columns(connection)
    add_employees_table(connection)
    return connection


def add_employees_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS employees (
            id TEXT PRIMARY KEY,
            organization_id TEXT NOT NULL,
            email TEXT NOT NULL,
            display_name TEXT NOT NULL DEFAULT '',
            employee_code TEXT NOT NULL DEFAULT '',
            department TEXT NOT NULL DEFAULT '',
            role_title TEXT NOT NULL DEFAULT '',
            joining_date TEXT NOT NULL DEFAULT '',
            work_location TEXT NOT NULL DEFAULT '',
            onboarded_at TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE (organization_id, email),
            FOREIGN KEY (organization_id) REFERENCES organizations(id)
        )
        """
    )
