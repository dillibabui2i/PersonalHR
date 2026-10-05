import re
import sqlite3

from app.rag.types import RetrievedChunk
from app.rag.vector_store import open_vector_database

NAME_WORD = r"(?!(?:leave|leaves)\b)[A-Za-z]+"
MULTI_NAME = rf"({NAME_WORD}(?:[ \t]+{NAME_WORD}){{0,3}})"
HEADING_LEAVE = re.compile(rf"^(?:\d+(?:\.\d+)*\.?\s+)?{MULTI_NAME}\s+leaves?$", re.IGNORECASE)
DAYS_OF_LEAVE = re.compile(rf"\b(\d+)\s+days of\s+{MULTI_NAME}\s+leaves?\b", re.IGNORECASE)
LEAVE_DAYS = re.compile(rf"\b(\d+)\s+{MULTI_NAME}\s+leaves?\s+days\b", re.IGNORECASE)
TITLED_LEAVE = re.compile(
    rf"^(?:\d+(?:\.\d+)*\.?\s+)?{MULTI_NAME}\s+leaves?(?:\s*\([A-Za-z]{{2,8}}\))?\s*[:\-–]\s*(\d+)\s+days\b",
    re.IGNORECASE | re.MULTILINE,
)
WORD_BEFORE_LEAVE = re.compile(r"\b([A-Za-z]{3,})\s+leaves?\b", re.IGNORECASE)
GENERIC_LEAVE_ASK = re.compile(r"\bhow many\s+leaves?\b", re.IGNORECASE)
CHOICE_LIMIT = 3


def display_leave(name: str) -> str:
    """Format a name the surrounding phrase already bounded, such as 'loss of pay'."""
    words = [
        word.lower()
        for word in re.findall(r"[A-Za-z]+", name)
        if word.lower() not in {"leave", "leaves"}
    ]
    while words and words[0] == "of":
        del words[0]
    while words and words[-1] == "of":
        del words[-1]
    if words == [] or len(words[0]) < 3:
        return ""
    return " ".join([words[0].capitalize(), *words[1:], "leave"])


def label_heads(label: str) -> list[str]:
    return [word for word in label.lower().split() if word not in {"", "leave", "leaves"}]


def matching_label(text: str, labels: list[str]) -> str:
    """Longest leave type from context whose own name appears in the text."""
    lowered = text.lower()
    best = ""
    best_size = 0
    for label in labels:
        heads = label_heads(label)
        if heads == []:
            continue
        phrase = r"\s+".join(re.escape(word) for word in heads)
        if re.search(rf"\b{phrase}\b", lowered) is None:
            continue
        if len(heads) <= best_size:
            continue
        best = " ".join(label.lower().split())
        best_size = len(heads)
    return best


def adjacent_leave(text: str) -> str:
    """The word directly before 'leave' when no contextual name is available."""
    if GENERIC_LEAVE_ASK.search(text) is not None:
        return ""
    found = WORD_BEFORE_LEAVE.search(text)
    if found is None:
        return ""
    return display_leave(found.group(1)).lower()


def mentioned_leave(text: str, labels: list[str] | None = None) -> str:
    """Leave type carried by context. Document and chat names win over a guess."""
    known = [] if labels is None else labels
    named = matching_label(text, known)
    if named != "":
        return named
    if known:
        return ""
    return adjacent_leave(text)


def prefer_leave_chunks(question: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Keep excerpts that name the asked leave type when retrieval also returned noise."""
    source = "\n".join(f"{chunk.section_title}\n{chunk.text}" for chunk in chunks)
    topic = mentioned_leave(question, labels_from_text(source)) or mentioned_leave(question)
    if topic == "" or chunks == []:
        return chunks
    matching = [
        chunk
        for chunk in chunks
        if source_has_leave_type(f"{chunk.section_title}\n{chunk.text}", topic)
    ]
    if len(matching) >= 2:
        return matching
    extras = [chunk for chunk in chunks if chunk not in matching]
    return matching + extras if matching else chunks


def source_has_leave_type(source: str, topic: str) -> bool:
    """True when the source names this leave type or its initials, not a stray first word."""
    cleaned = topic.strip()
    if cleaned == "" or source.strip() == "":
        return False
    if re.search(rf"\b{re.escape(cleaned)}\b", source, re.IGNORECASE) is not None:
        return True
    words = [word for word in cleaned.lower().split() if word != ""]
    if words and words[-1] == "leave":
        heads = r"\s+".join(re.escape(word) for word in words[:-1])
        phrase = rf"{heads}\s+leaves?" if heads else r"leaves?"
        if re.search(rf"\b{phrase}\b", source, re.IGNORECASE) is not None:
            return True
    code = leave_initials(cleaned)
    return len(code) >= 2 and re.search(rf"\b{re.escape(code)}\b", source) is not None


def leave_initials(label: str) -> str:
    return "".join(word[0] for word in label.split() if word != "").upper()


def leave_mentions(text: str, labels: list[str] | None = None) -> list[str]:
    known = [] if labels is None else labels
    found: list[str] = []
    seen: set[str] = set()
    for label in known:
        named = matching_label(text, [label])
        if named == "" or named in seen:
            continue
        seen.add(named)
        found.append(named)
    if known:
        return found
    for match in WORD_BEFORE_LEAVE.finditer(text):
        window = text[max(0, match.start() - 12) : match.end()]
        if GENERIC_LEAVE_ASK.search(window) is not None:
            continue
        label = display_leave(match.group(1))
        if label == "" or label.lower() in seen:
            continue
        seen.add(label.lower())
        found.append(label)
    return found


def leave_for_code(text: str, labels: list[str]) -> str:
    """Match a short form such as 'PL' to one leave type already named in the documents."""
    named = mentioned_leave(text, labels)
    if named != "":
        return named
    for token in re.findall(r"\b[A-Za-z]{2,8}\b", text):
        code = token.upper()
        matches: list[str] = []
        for label in labels:
            if leave_initials(label) == code and label.lower() not in matches:
                matches.append(label.lower())
        if len(matches) == 1:
            return matches[0]
    return ""


def replace_leave_code(text: str, label: str) -> str:
    code = leave_initials(label)
    if code == "":
        return text
    replaced, count = re.subn(rf"\b{re.escape(code)}\s+leaves?\b", label, text, count=1, flags=re.IGNORECASE)
    if count == 0:
        replaced, count = re.subn(rf"\b{re.escape(code)}\b", label, text, count=1, flags=re.IGNORECASE)
    if count == 0:
        return text
    return replaced


def hinted_leave(text: str, labels: list[str]) -> str:
    """Map a short follow-up such as 'and for sick?' onto one leave type from context."""
    return mentioned_leave(text, labels)


def _remember(found: list[tuple[str, int]], seen: set[str], label: str, days: int) -> None:
    if label == "" or label.lower() in seen:
        return
    seen.add(label.lower())
    found.append((label, days))


def leave_allowances(text: str) -> list[tuple[str, int]]:
    """Leave types and day counts stated in document text, in the order they appear."""
    matches: list[tuple[int, str, int]] = []
    for pattern, name_group, days_group in (
        (DAYS_OF_LEAVE, 2, 1),
        (LEAVE_DAYS, 2, 1),
        (TITLED_LEAVE, 1, 2),
    ):
        for match in pattern.finditer(text):
            label = display_leave(match.group(name_group))
            if label == "":
                continue
            matches.append((match.start(), label, int(match.group(days_group))))
    found: list[tuple[str, int]] = []
    seen: set[str] = set()
    for _start, label, days in sorted(matches, key=lambda item: item[0]):
        _remember(found, seen, label, days)
    return found


def leave_headings(text: str) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        match = HEADING_LEAVE.match(line.strip())
        if match is None:
            continue
        label = display_leave(match.group(1))
        if label == "" or label.lower() in seen:
            continue
        seen.add(label.lower())
        labels.append(label)
    return labels


def differing_leave_choices(allowances: list[tuple[str, int]]) -> list[str]:
    """Button labels when the documents count two or more leave types differently."""
    counts = {days for _label, days in allowances}
    if len(allowances) < 2 or len(counts) < 2:
        return []
    return [label for label, _days in allowances[:CHOICE_LIMIT]]


def organization_document_text(organization_id: str) -> str:
    if organization_id.strip() == "":
        return ""
    try:
        connection = open_vector_database(organization_id)
    except ValueError:
        return ""
    try:
        rows = connection.execute("SELECT section_title, chunk_text FROM chunks_fts").fetchall()
    except sqlite3.Error:
        return ""
    finally:
        connection.close()
    return "\n".join(f"{row['section_title']}\n{row['chunk_text']}" for row in rows)


def labels_from_text(text: str) -> list[str]:
    """Leave names stated in this text, from its headings and day counts."""
    return labels_in_text(text, leave_allowances(text))


def labels_in_text(text: str, allowances: list[tuple[str, int]]) -> list[str]:
    labels = [label for label, _days in allowances]
    seen = {label.lower() for label in labels}
    for label in leave_headings(text):
        if label.lower() in seen:
            continue
        seen.add(label.lower())
        labels.append(label)
    return labels


def document_leave_context(organization_id: str) -> tuple[list[str], list[str]]:
    """Leave names in the uploaded documents, and button labels when their day counts differ."""
    text = organization_document_text(organization_id)
    allowances = leave_allowances(text)
    return labels_in_text(text, allowances), differing_leave_choices(allowances)


def document_leave_choices(organization_id: str) -> list[str]:
    _labels, choices = document_leave_context(organization_id)
    return choices
