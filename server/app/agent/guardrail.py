import re

from app.agent.clarify import labels_from_text, leave_allowances, mentioned_leave, source_has_leave_type
from app.agent.planner import NOTE_ASK
from app.rag.prompts import (
    CONVERSATION_MISS,
    MEMORY_DISABLED_REPLY,
    NO_ANSWER,
    PORTAL_DISABLED_REPLY,
    UNAVAILABLE_ANSWER,
)
from app.rag.types import Citation, RagAnswer, RetrievedChunk, citation_for_chunk

NUMBER = re.compile(r"\d+(?:\.\d+)?")
LIST_MARKER = re.compile(r"(?m)^\s*\d+[.)]\s+")
UNAVAILABLE_LINE = re.compile(r"\bunavailable\b", re.IGNORECASE)
PAGE_UNAVAILABLE = "The portal could not be opened."
ALLOWED_WITHOUT_FINDINGS = {
    NO_ANSWER,
    UNAVAILABLE_ANSWER,
    PORTAL_DISABLED_REPLY,
    MEMORY_DISABLED_REPLY,
    CONVERSATION_MISS,
}
PASSED_REASON = "Citations and numbers match the sources read this turn."
ENTITLEMENT_ASK = re.compile(
    r"\b(how many|how much|entitled|entitlement|allowance|quota)\b",
    re.IGNORECASE,
)
QUARTERLY_PERIOD = re.compile(
    r"\b(\d+)\s+days\b.{0,40}\b(?:per quarter|each quarter|quarterly)\b",
    re.IGNORECASE,
)
ANNUAL_PERIOD = re.compile(
    r"\b(\d+)\s+days\b.{0,60}\b(?:calendar year|per year|annually|in a year|a year)\b|"
    r"\bentitled to\s+(\d+)\s+days\b",
    re.IGNORECASE,
)


def answer_numbers(answer: str) -> list[str]:
    without_lists = LIST_MARKER.sub("", answer)
    found: list[str] = []
    for number in NUMBER.findall(without_lists):
        if number not in found:
            found.append(number)
    return found


def number_in_source(number: str, source: str) -> bool:
    return re.search(rf"(?<!\d){re.escape(number)}(?!\d)", source) is not None


def citation_in_excerpts(citation: Citation, chunks: list[RetrievedChunk]) -> bool:
    name = citation.document_name.strip().lower()
    section = citation.section_title.strip().lower()
    return any(
        chunk.document_name.strip().lower() == name and chunk.section_title.strip().lower() == section
        for chunk in chunks
    )


def page_answered(page_facts: str | None) -> bool:
    if page_facts is None:
        return False
    text = page_facts.strip()
    return text != "" and text != PAGE_UNAVAILABLE


def tool_returned_findings(tool_text: str) -> bool:
    for line in tool_text.splitlines():
        cleaned = line.strip()
        if cleaned == "" or UNAVAILABLE_LINE.search(cleaned):
            continue
        return True
    return False


def source_text(
    chunks: list[RetrievedChunk],
    page_text: str | None,
    page_facts: str | None,
    tool_text: str,
    employee_facts: str = "",
) -> str:
    parts = [f"{chunk.document_name}\n{chunk.section_title}\n{chunk.text}" for chunk in chunks]
    if page_text:
        parts.append(page_text)
    if page_facts:
        parts.append(page_facts)
    if tool_text.strip() != "":
        parts.append(tool_text)
    if employee_facts.strip() != "":
        parts.append(employee_facts)
    return "\n".join(parts)


def source_mentions(source: str, word: str) -> bool:
    if re.search(rf"\b{re.escape(word)}\b", source, re.IGNORECASE) is not None:
        return True
    return len(word) >= 4 and re.search(rf"\b{re.escape(word)}", source, re.IGNORECASE) is not None


def missing_leave_claim(answer: str, sources: str) -> str | None:
    """Reject a number tied to a leave type the sources never name."""
    sentences = re.split(r"(?<=[.!?])\s+", answer)
    labels = labels_from_text(sources)
    for sentence in sentences:
        numbers = answer_numbers(sentence)
        if numbers == []:
            continue
        topic = mentioned_leave(sentence, labels) or mentioned_leave(sentence)
        if topic == "":
            continue
        if source_has_leave_type(sources, topic) or source_mentions(sources, topic.split()[0]):
            continue
        core = " ".join(topic.split()[-2:]) if len(topic.split()) > 1 else topic
        if core != topic and source_has_leave_type(sources, core):
            continue
        return (
            f"The number {numbers[0]} is attached to {topic}, which is not in the policy excerpts "
            "or the date counts."
        )
    return None


def allowance_citations(chunks: list[RetrievedChunk], allowances: list[tuple[str, int]]) -> list[Citation]:
    citations: list[Citation] = []
    for label, days in allowances:
        head = label.split()[0]
        for chunk in chunks:
            if not source_mentions(chunk.text, head) or not number_in_source(str(days), chunk.text):
                continue
            citation = citation_for_chunk(chunk)
            if citation not in citations:
                citations.append(citation)
            break
    return citations


def chunk_allowances(chunks: list[RetrievedChunk]) -> list[tuple[str, int]]:
    found: list[tuple[str, int]] = []
    seen: set[str] = set()
    for chunk in chunks:
        for label, days in leave_allowances(f"{chunk.section_title}\n{chunk.text}"):
            if label.lower() in seen:
                continue
            seen.add(label.lower())
            found.append((label, days))
    return found


def allowance_sentence(allowances: list[tuple[str, int]]) -> str:
    parts = [f"{days} days of {label}" for label, days in allowances[:3]]
    if parts == []:
        return ""
    if len(parts) == 1:
        listed = parts[0]
    elif len(parts) == 2:
        listed = f"{parts[0]} and {parts[1]}"
    else:
        listed = f"{', '.join(parts[:-1])}, and {parts[-1]}"
    return f" I found {listed}."


def entitlement_question(question: str, labels: list[str] | None = None) -> bool:
    return ENTITLEMENT_ASK.search(question) is not None and (
        mentioned_leave(question, labels) or mentioned_leave(question)
    ) != ""


def period_days(pattern: re.Pattern[str], text: str) -> set[int]:
    found: set[int] = set()
    for match in pattern.finditer(text):
        for group in match.groups():
            if group is not None:
                found.add(int(group))
    return found


def matching_leave_allowances(
    question: str,
    chunks: list[RetrievedChunk],
) -> list[tuple[str, int, RetrievedChunk]]:
    source = "\n".join(f"{chunk.section_title}\n{chunk.text}" for chunk in chunks)
    topic = mentioned_leave(question, labels_from_text(source)) or mentioned_leave(question)
    if topic == "" or chunks == []:
        return []
    annual: list[tuple[str, int, RetrievedChunk]] = []
    other: list[tuple[str, int, RetrievedChunk]] = []
    seen: set[tuple[str, int, str]] = set()
    for chunk in chunks:
        text = f"{chunk.section_title}\n{chunk.text}"
        quarterly = period_days(QUARTERLY_PERIOD, text)
        yearly = period_days(ANNUAL_PERIOD, text)
        for label, days in leave_allowances(text):
            if label.lower() != topic and topic not in label.lower():
                continue
            if days in quarterly and days not in yearly:
                continue
            key = (label.lower(), days, chunk.document_name)
            if key in seen:
                continue
            seen.add(key)
            item = (label, days, chunk)
            if days in yearly:
                annual.append(item)
            else:
                other.append(item)
    return annual if annual else other


def leave_count_answer(question: str, chunks: list[RetrievedChunk]) -> RagAnswer | None:
    """Answer a how-many leave question from excerpts when the composer refuses."""
    matches = matching_leave_allowances(question, chunks)
    if matches == []:
        return None
    citations = []
    for _label, _days, chunk in matches:
        citation = citation_for_chunk(chunk)
        if citation not in citations:
            citations.append(citation)
    counts = {days for _label, days, _chunk in matches}
    topic = matches[0][0].lower()
    if len(counts) == 1:
        days = next(iter(counts))
        return RagAnswer(
            answer=f"You get **{days} days** of {topic} per calendar year.",
            citations=citations,
        )
    lines = [f"Your organization's documents list more than one {topic} entitlement:"]
    for _label, days, chunk in matches:
        lines.append(f"- **{days} days** in {chunk.document_name}")
    lines.append("I cannot tell which document is current, so I am showing both.")
    return RagAnswer(answer="\n".join(lines), citations=citations)


def leave_gap_answer(question: str, chunks: list[RetrievedChunk]) -> RagAnswer | None:
    """Say a named leave type is absent instead of relabeling a different allowance."""
    source = "\n".join(f"{chunk.section_title}\n{chunk.text}" for chunk in chunks)
    topic = mentioned_leave(question, labels_from_text(source)) or mentioned_leave(question)
    if topic == "" or chunks == []:
        return None
    if source_has_leave_type(source, topic):
        return None
    if NOTE_ASK.search(question) is not None:
        return RagAnswer(
            answer=f"I couldn't find a medical note rule for {topic} in your organization's documents.",
            citations=[],
        )
    allowances = chunk_allowances(chunks)
    reply = f"I couldn't find a {topic} allowance in your organization's documents.{allowance_sentence(allowances)}"
    return RagAnswer(answer=reply, citations=allowance_citations(chunks, allowances))


def grounding_failure(
    answer: str,
    citations: list[Citation],
    chunks: list[RetrievedChunk],
    page_text: str | None,
    page_facts: str | None,
    tool_text: str,
    employee_facts: str = "",
) -> str | None:
    """Return a reason when the answer cites an unread section or states an unseen number."""
    reasons: list[str] = []
    sources = source_text(chunks, page_text, page_facts, tool_text, employee_facts)
    for number in answer_numbers(answer):
        if not number_in_source(number, sources):
            reasons.append(
                f"The number {number} was not in the documents, page text, or tool results read this turn."
            )
    claim = missing_leave_claim(answer, sources)
    if claim is not None:
        reasons.append(claim)
    for citation in citations:
        if not citation_in_excerpts(citation, chunks):
            reasons.append(
                f"The citation {citation.document_name} — {citation.section_title} was not returned by search this turn."
            )
    if (
        chunks == []
        and not page_answered(page_facts)
        and not tool_returned_findings(tool_text)
        and employee_facts.strip() == ""
        and answer.strip() not in ALLOWED_WITHOUT_FINDINGS
    ):
        reasons.append("No document excerpt or page text answered this question.")
    if reasons == []:
        return None
    return " ".join(reasons)
