import re

from app.rag.prompts import NO_ANSWER, UNAVAILABLE_ANSWER
from app.rag.vector_store import section_titles

SUGGESTION_LIMIT = 3
NUMBERED_TITLE = re.compile(r"^(?:\d+(?:\.\d+)*\.?\s+)+")
LEADING_DATE = re.compile(r"^\d{1,2}[-/][A-Za-z]{3}[-/]\d{2,4}\b")
DANGLING_WORD = {"and", "or", "of", "the", "to", "for", "with"}


def topic_label(title: str) -> str:
    """Keep a short heading and drop numbered prefixes and sentence fragments."""
    cleaned = " ".join(title.split()).strip(" .")
    label = NUMBERED_TITLE.sub("", cleaned).strip(" .")
    words = label.split()
    if label == "" or len(label) > 60 or len(words) > 8 or label.endswith((":", ";", ",")):
        return ""
    if len(words) == 1 and len(words[0]) < 6:
        return ""
    if LEADING_DATE.match(label) is not None or words[-1].lower() in DANGLING_WORD:
        return ""
    return label


def question_for_topic(label: str) -> str:
    return f"What does the policy say about {label}?"


def questions_for_titles(titles: list[str], limit: int, skip: set[str] | None = None) -> tuple[str, ...]:
    ignored = set() if skip is None else {item.lower() for item in skip}
    questions: list[str] = []
    for title in titles:
        label = topic_label(title)
        if label == "" or label.lower() in ignored:
            continue
        question = question_for_topic(label)
        if question.lower() in ignored or question in questions:
            continue
        questions.append(question)
        if len(questions) == limit:
            break
    return tuple(questions)


def follow_up_questions(organization_id: str, document_ids: frozenset[str], question: str, answer: str) -> tuple[str, ...]:
    if answer.strip() in {NO_ANSWER, UNAVAILABLE_ANSWER, ""}:
        return ()
    titles = section_titles(organization_id, document_ids)
    return questions_for_titles(titles, SUGGESTION_LIMIT, {question})
