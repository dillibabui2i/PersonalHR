import json
import sqlite3
import uuid

import sqlite_vec

from app.core.database import data_directory
from app.core.settings import load_settings
from app.rag.types import Chunk, RetrievedChunk


def open_vector_database(organization_id: str) -> sqlite3.Connection:
    organization_key = str(uuid.UUID(organization_id))
    vector_directory = data_directory() / "vectors"
    vector_directory.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(vector_directory / f"{organization_key}.db")
    connection.row_factory = sqlite3.Row
    connection.enable_load_extension(True)
    sqlite_vec.load(connection)
    connection.enable_load_extension(False)
    from app.organizations.models import organization_embedding

    _model, dimensions = organization_embedding(organization_id)
    if dimensions <= 0:
        dimensions = int(load_settings().embed_dimensions)
    connection.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING vec0(
            chunk_id text primary key,
            document_id text,
            document_name text,
            section_title text,
            chunk_text text,
            embedding float[{dimensions}]
        )
        """
    )
    connection.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            chunk_id UNINDEXED,
            document_id UNINDEXED,
            document_name UNINDEXED,
            section_title,
            chunk_text
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS terms (
            document_id TEXT NOT NULL,
            short_form TEXT NOT NULL,
            long_form TEXT NOT NULL
        )
        """
    )
    return connection


def chunk_from_row(row: sqlite3.Row) -> RetrievedChunk:
    keys = row.keys()
    return RetrievedChunk(
        chunk_id=str(row["chunk_id"]),
        document_id=str(row["document_id"]),
        document_name=str(row["document_name"]),
        section_title=str(row["section_title"]),
        text=str(row["chunk_text"]),
        vector_distance=float(row["distance"]) if "distance" in keys else None,
    )


def nearest_chunks(organization_id: str, vector: list[float], limit: int) -> list[RetrievedChunk]:
    with open_vector_database(organization_id) as connection:
        rows = connection.execute(
            """
            SELECT chunk_id, document_id, document_name, section_title, chunk_text, distance
            FROM chunks
            WHERE embedding MATCH ?
              AND k = ?
            ORDER BY distance
            """,
            (json.dumps(vector), limit),
        ).fetchall()
    return [chunk_from_row(row) for row in rows]


def keyword_chunks(organization_id: str, match: str, limit: int) -> list[RetrievedChunk]:
    if match == "":
        return []
    with open_vector_database(organization_id) as connection:
        try:
            rows = connection.execute(
                """
                SELECT chunk_id, document_id, document_name, section_title, chunk_text
                FROM chunks_fts
                WHERE chunks_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (match, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
    return [chunk_from_row(row) for row in rows]


def section_titles(organization_id: str, document_ids: frozenset[str]) -> list[str]:
    if organization_id.strip() == "" or document_ids == frozenset():
        return []
    marks = ",".join("?" for _ in document_ids)
    try:
        connection = open_vector_database(organization_id)
    except ValueError:
        return []
    try:
        rows = connection.execute(
            f"""
            SELECT section_title
            FROM chunks_fts
            WHERE document_id IN ({marks})
            ORDER BY rowid
            """,
            tuple(sorted(document_ids)),
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        connection.close()
    titles: list[str] = []
    seen: set[str] = set()
    for row in rows:
        title = " ".join(str(row["section_title"]).split())
        if title == "" or title.lower() in seen:
            continue
        seen.add(title.lower())
        titles.append(title)
    return titles


def read_terms(organization_id: str) -> dict[str, str]:
    with open_vector_database(organization_id) as connection:
        rows = connection.execute("SELECT short_form, long_form FROM terms").fetchall()
    terms: dict[str, str] = {}
    for row in rows:
        terms.setdefault(str(row["short_form"]), str(row["long_form"]))
    return terms


def delete_document_index(organization_id: str, document_id: str) -> None:
    with open_vector_database(organization_id) as connection:
        connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        connection.execute("DELETE FROM chunks_fts WHERE document_id = ?", (document_id,))
        connection.execute("DELETE FROM terms WHERE document_id = ?", (document_id,))


def replace_document_chunks(
    organization_id: str,
    document_id: str,
    document_name: str,
    chunks: list[Chunk],
    embeddings: list[list[float]],
    terms: dict[str, str],
) -> None:
    with open_vector_database(organization_id) as connection:
        connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        connection.execute("DELETE FROM chunks_fts WHERE document_id = ?", (document_id,))
        connection.execute("DELETE FROM terms WHERE document_id = ?", (document_id,))
        for short_form, long_form in terms.items():
            connection.execute(
                "INSERT INTO terms (document_id, short_form, long_form) VALUES (?, ?, ?)",
                (document_id, short_form, long_form),
            )
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            chunk_id = str(uuid.uuid4())
            connection.execute(
                """
                INSERT INTO chunks (
                    chunk_id, document_id, document_name, section_title, chunk_text, embedding
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (chunk_id, document_id, document_name, chunk.section_title, chunk.text, json.dumps(embedding)),
            )
            connection.execute(
                """
                INSERT INTO chunks_fts (
                    chunk_id, document_id, document_name, section_title, chunk_text
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (chunk_id, document_id, document_name, chunk.section_title, chunk.text),
            )
