import logging
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from fastapi import HTTPException

from app.cache.service import invalidate_organization_cache
from app.core.database import data_directory, open_app_database
from app.core.settings import load_settings
from app.llm.embeddings import EmbeddingFailed
from app.organizations.service import read_organization
from app.rag.ingestion.loaders import SUPPORTED_SUFFIXES
from app.rag.pipeline import ingest_document
from app.rag.vector_store import delete_document_index

request_log = logging.getLogger("personal_hr")
ALLOWED_SUFFIXES = SUPPORTED_SUFFIXES


@dataclass(frozen=True)
class DocumentRecord:
    id: str
    file_name: str
    status: str
    error_message: str


def document_from_row(row: sqlite3.Row) -> DocumentRecord:
    return DocumentRecord(
        id=row["id"],
        file_name=row["file_name"],
        status=row["status"],
        error_message=row["error_message"],
    )


def list_documents(organization_id: str) -> list[DocumentRecord]:
    read_organization(organization_id)
    with open_app_database() as connection:
        rows = connection.execute(
            """
            SELECT id, file_name, status, error_message
            FROM documents
            WHERE organization_id = ?
            ORDER BY created_at
            """,
            (organization_id,),
        ).fetchall()
    return [document_from_row(row) for row in rows]


def ready_document_ids(organization_id: str) -> frozenset[str]:
    with open_app_database() as connection:
        rows = connection.execute(
            "SELECT id FROM documents WHERE organization_id = ? AND status = ?",
            (organization_id, "ready"),
        ).fetchall()
    return frozenset(str(row["id"]) for row in rows)


def stored_file(stored_path: str) -> Path:
    root = data_directory().resolve()
    path = (root / stored_path).resolve()
    if not path.is_relative_to(root):
        raise HTTPException(status_code=400, detail="The file could not be read.")
    return path


def save_uploaded_document(organization_id: str, file_name: str, content: bytes) -> DocumentRecord:
    read_organization(organization_id)
    cleaned_name = Path(file_name).name.strip()
    if cleaned_name == "":
        raise HTTPException(status_code=400, detail="Choose a file to upload.")
    suffix = Path(cleaned_name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail="Upload a PDF, DOCX, or TXT file.")
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="The file is empty.")
    limit_megabytes = load_settings().max_upload_megabytes
    if len(content) > limit_megabytes * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"Upload a file that is {limit_megabytes} MB or smaller.")

    document_id = str(uuid.uuid4())
    folder = data_directory() / "files" / organization_id
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / f"{document_id}{suffix}"
    destination.write_bytes(content)
    stored_path = destination.resolve().relative_to(data_directory().resolve()).as_posix()
    created_at = datetime.now(timezone.utc).isoformat()
    with open_app_database() as connection:
        connection.execute(
            """
            INSERT INTO documents (
                id, organization_id, file_name, stored_path, status, error_message, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (document_id, organization_id, cleaned_name, stored_path, "pending", "", created_at),
        )
    invalidate_organization_cache(organization_id)
    return DocumentRecord(
        id=document_id,
        file_name=cleaned_name,
        status="pending",
        error_message="",
    )


def set_document_status(document_id: str, status: str, error_message: str) -> None:
    organization_id = ""
    with open_app_database() as connection:
        row = connection.execute(
            "SELECT organization_id FROM documents WHERE id = ?",
            (document_id,),
        ).fetchone()
        if row is not None:
            organization_id = str(row["organization_id"])
        connection.execute(
            """
            UPDATE documents
            SET status = ?, error_message = ?
            WHERE id = ?
            """,
            (status, error_message, document_id),
        )
    if organization_id != "":
        invalidate_organization_cache(organization_id)


def read_document_row(document_id: str) -> tuple[str, str, str, str] | None:
    with open_app_database() as connection:
        row = connection.execute(
            """
            SELECT organization_id, file_name, stored_path, status
            FROM documents
            WHERE id = ?
            """,
            (document_id,),
        ).fetchone()
        if row is None:
            return None
        return (row["organization_id"], row["file_name"], row["stored_path"], row["status"])


def delete_document(organization_id: str, document_id: str) -> None:
    read_organization(organization_id)
    row = read_document_row(document_id)
    if row is None or row[0] != organization_id:
        raise HTTPException(status_code=404, detail="That document was not found.")
    delete_document_index(organization_id, document_id)
    path = stored_file(row[2])
    if path.is_file():
        path.unlink()
    with open_app_database() as connection:
        connection.execute(
            "DELETE FROM documents WHERE id = ? AND organization_id = ?",
            (document_id, organization_id),
        )
    invalidate_organization_cache(organization_id)


def embed_document(organization_id: str, document_id: str) -> None:
    row = read_document_row(document_id)
    if row is None or row[0] != organization_id:
        return
    if row[3] != "pending":
        return
    try:
        ingest_document(organization_id, document_id, row[1], stored_file(row[2]))
    except EmbeddingFailed as error:
        request_log.info("embed failed for %s: %s", document_id, error.detail)
        set_document_status(document_id, "failed", error.detail)
        return
    except ValueError as error:
        detail = str(error) if str(error) != "" else "The file could not be read."
        request_log.info("embed failed for %s: %s", document_id, detail)
        set_document_status(document_id, "failed", detail)
        return
    except Exception:
        request_log.exception("embed failed for %s", document_id)
        set_document_status(document_id, "failed", "The file could not be embedded.")
        return
    set_document_status(document_id, "ready", "")
