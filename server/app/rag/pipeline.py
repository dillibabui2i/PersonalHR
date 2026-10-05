from pathlib import Path

from app.llm.embeddings import embed_texts
from app.organizations.models import organization_embedding
from app.rag.glossary import find_abbreviations
from app.rag.ingestion.chunking import build_chunks
from app.rag.ingestion.loaders import load_text
from app.rag.vector_store import replace_document_chunks


def ingest_document(organization_id: str, document_id: str, document_name: str, path: Path) -> None:
    text = load_text(path)
    model, dimensions = organization_embedding(organization_id)
    chunks = build_chunks(text, Path(document_name).stem, model, dimensions)
    if chunks == []:
        raise ValueError("The file has no text to embed.")
    embeddings = embed_texts(
        [f"{chunk.section_title}\n{chunk.text}" for chunk in chunks],
        "document",
        model,
        dimensions,
    )
    replace_document_chunks(
        organization_id,
        document_id,
        document_name,
        chunks,
        embeddings,
        find_abbreviations(text),
    )
