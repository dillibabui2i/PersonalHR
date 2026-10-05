from pathlib import Path

from docx import Document
from pypdf import PdfReader

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt"}


def load_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("The text file could not be read.") from error
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix == ".docx":
        document = Document(str(path))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)
    raise ValueError("Upload a PDF, DOCX, or TXT file.")
