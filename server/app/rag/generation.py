import re
from collections.abc import Callable
from datetime import datetime

from app.activity.trace import TurnTrace, open_step, raise_if_cancelled
from app.chat.cancel import TurnCancelled
from app.llm.chat_model import ChatModelFailed, generate_json, stream_json
from app.rag.prompts import (
    ANSWER_PROMPT,
    ANSWER_SCHEMA,
    BRIEF_STYLE,
    DATE_INSTRUCTIONS,
    DETAILED_STYLE,
    GENERAL_PROMPT,
    GENERAL_SCHEMA,
    GREETING_PROMPT,
    CONVERSATION_INSTRUCTIONS,
    MEMORY_INSTRUCTIONS,
    NO_ANSWER,
    PAGE_INSTRUCTIONS,
    UNAVAILABLE_ANSWER,
)
from app.rag.types import Citation, RagAnswer, RetrievedChunk, citation_for_chunk, excerpt_text

SOURCE_MARKER = re.compile(r"\s*\[\d+(?:\s*,\s*\d+)*\]")


def clean_answer(answer: str) -> str:
    cleaned = SOURCE_MARKER.sub("", answer).replace("\r\n", "\n").replace("\r", "\n")
    lines = [" ".join(line.split()) for line in cleaned.split("\n")]
    return "\n".join(lines).strip()


def format_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{index}] documentName: {chunk.document_name}\n"
        f"sectionTitle: {chunk.section_title}\n"
        f"text: {chunk.text}"
        for index, chunk in enumerate(chunks, start=1)
    )


def used_citations(raw_sources: object, chunks: list[RetrievedChunk]) -> list[Citation]:
    if not isinstance(raw_sources, list):
        return []
    grouped: dict[tuple[str, str], list[str]] = {}
    order: list[tuple[str, str]] = []
    for source in raw_sources:
        if not isinstance(source, int) or source < 1 or source > len(chunks):
            continue
        chunk = chunks[source - 1]
        key = (chunk.document_name, chunk.section_title)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        if chunk.text not in grouped[key]:
            grouped[key].append(chunk.text)
    return [
        Citation(document_name=name, section_title=section, excerpt=excerpt_text("\n".join(grouped[(name, section)])))
        for name, section in order
    ]


ANSWERED_TRUE = re.compile(r'"answered"\s*:\s*true')
TRUTHY_ANSWERED = {True, "true", "True", 1, "1"}


def answer_is_showable(buffer: str) -> bool:
    return ANSWERED_TRUE.search(buffer) is not None


def model_answered(parsed: dict[str, object], reply: str) -> bool:
    if reply == "" or reply == NO_ANSWER:
        return False
    if parsed.get("answered") in TRUTHY_ANSWERED:
        return True
    return reply != ""


def complete_json(
    prompt: str,
    schema: dict[str, object],
    trace: TurnTrace | None,
    ready_to_show: Callable[[str], bool] | None = None,
) -> dict[str, object]:
    raise_if_cancelled(trace)
    if trace is not None and trace.on_token is not None:
        return stream_json(prompt, schema, 120, trace.on_token, trace.cancel, ready_to_show)
    return generate_json(prompt, schema, timeout_seconds=120)


def generate_general_answer(question: str, trace: TurnTrace | None = None) -> RagAnswer:
    now = datetime.now().astimezone().strftime("%A, %d %B %Y, %I:%M %p %Z")
    prompt = GENERAL_PROMPT.format(now=now, question=question.strip())
    with open_step(trace, "general answer", question.strip()) as answering:
        try:
            parsed = complete_json(prompt, GENERAL_SCHEMA, trace)
        except TurnCancelled:
            answering.status = "stopped"
            answering.output_summary = "Stopped"
            raise
        except ChatModelFailed:
            answering.status = "failed"
            answering.output_summary = UNAVAILABLE_ANSWER
            return RagAnswer(answer=UNAVAILABLE_ANSWER, citations=[])
        answer = parsed.get("answer", "")
        if not isinstance(answer, str) or clean_answer(answer) == "":
            answering.status = "failed"
            answering.output_summary = UNAVAILABLE_ANSWER
            return RagAnswer(answer=UNAVAILABLE_ANSWER, citations=[])
        reply = clean_answer(answer)
        answering.output_summary = reply
        return RagAnswer(answer=reply, citations=[])


def generate_greeting(question: str, employee_facts: str = "", trace: TurnTrace | None = None) -> RagAnswer:
    """Write a short conversational reply. This is not taken from a greeting template."""
    now = datetime.now().astimezone().strftime("%A, %d %B %Y, %I:%M %p %Z")
    facts = employee_facts.strip() or "none"
    prompt = GREETING_PROMPT.format(now=now, facts=facts, message=question.strip())
    with open_step(trace, "greeting", question.strip()) as answering:
        try:
            parsed = complete_json(prompt, GENERAL_SCHEMA, trace)
        except TurnCancelled:
            answering.status = "stopped"
            answering.output_summary = "Stopped"
            raise
        except ChatModelFailed:
            answering.status = "failed"
            answering.output_summary = UNAVAILABLE_ANSWER
            return RagAnswer(answer=UNAVAILABLE_ANSWER, citations=[])
        answer = parsed.get("answer", "")
        if not isinstance(answer, str) or clean_answer(answer) == "":
            answering.status = "failed"
            answering.output_summary = UNAVAILABLE_ANSWER
            return RagAnswer(answer=UNAVAILABLE_ANSWER, citations=[])
        reply = clean_answer(answer)
        answering.output_summary = reply
        return RagAnswer(answer=reply, citations=[])


def style_instructions(answer_style: str) -> str:
    if answer_style == "detailed":
        return DETAILED_STYLE
    return BRIEF_STYLE


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    page_text: str | None = None,
    date_note: str = "",
    trace: TurnTrace | None = None,
    record_step: bool = True,
    correction: str = "",
    employee_facts: str = "",
    answer_style: str = "brief",
    conversation: str = "",
) -> RagAnswer:
    raise_if_cancelled(trace)
    del page_text
    if chunks == [] and date_note == "" and employee_facts == "" and conversation == "":
        return RagAnswer(answer=NO_ANSWER, citations=[])
    page_block = ""
    page_instructions = PAGE_INSTRUCTIONS
    date_block = ""
    date_instructions = ""
    today = datetime.now().astimezone().strftime("%A, %d %B %Y")
    combined_dates = f"Today is {today}." if date_note == "" else f"Today is {today}.\n{date_note}"
    date_block = f"Date arithmetic:\n{combined_dates}\n\n"
    date_instructions = DATE_INSTRUCTIONS
    memory_block = ""
    memory_instructions = ""
    if employee_facts != "":
        memory_block = f"Employee-stated facts:\n{employee_facts}\n\n"
        memory_instructions = MEMORY_INSTRUCTIONS
    conversation_block = ""
    conversation_instructions = ""
    if conversation != "":
        conversation_block = f"Earlier messages in this conversation:\n{conversation}\n\n"
        conversation_instructions = CONVERSATION_INSTRUCTIONS
    context = format_context(chunks) if chunks != [] else "none"
    prompt = ANSWER_PROMPT.format(
        question=question.strip(),
        style_instructions=style_instructions(answer_style),
        page_instructions=page_instructions,
        date_instructions=date_instructions,
        memory_instructions=memory_instructions,
        conversation_instructions=conversation_instructions,
        page_block=page_block,
        date_block=date_block,
        memory_block=memory_block,
        conversation_block=conversation_block,
        context=context,
    )
    if correction != "":
        prompt += (
            "\n\nThe previous answer was rejected:\n"
            f"{correction}\n"
            "Answer again. Every number must appear in the documents, the question, tool results, "
            "earlier messages, or employee facts above. "
            "Every citation must be one of the excerpts above. "
            "If you cannot, set answered to false and leave answer empty.\n"
        )
    step_trace = trace if record_step else None
    with open_step(step_trace, "answer", question.strip()) as answering:
        try:
            parsed = complete_json(prompt, ANSWER_SCHEMA, trace, answer_is_showable)
        except TurnCancelled:
            answering.status = "stopped"
            answering.output_summary = "Stopped"
            raise
        except ChatModelFailed:
            answering.status = "failed"
            answering.output_summary = UNAVAILABLE_ANSWER
            return RagAnswer(answer=UNAVAILABLE_ANSWER, citations=[])
        answer = parsed.get("answer", "")
        reply = clean_answer(answer) if isinstance(answer, str) else ""
        if not model_answered(parsed, reply):
            answering.output_summary = NO_ANSWER
            return RagAnswer(answer=NO_ANSWER, citations=[])
        answering.output_summary = reply
        citations = used_citations(parsed.get("sources"), chunks)
        if citations == [] and chunks != []:
            citations = [citation_for_chunk(chunks[0])]
        return RagAnswer(answer=reply, citations=citations)
