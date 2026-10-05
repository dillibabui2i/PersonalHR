import re

ABBREVIATION_PATTERN = re.compile(r"\b((?:[A-Z][A-Za-z]+[\s-]+){1,5}[A-Z][A-Za-z]+)\s*\(([A-Z]{2,8})\)")
WORD_SEPARATOR = re.compile(r"[\s-]+")


def initials_match(long_form: str, short_form: str) -> bool:
    initials = "".join(word[0] for word in WORD_SEPARATOR.split(long_form) if word != "")
    return initials.upper().endswith(short_form.upper())


def find_abbreviations(text: str) -> dict[str, str]:
    found: dict[str, str] = {}
    flattened = re.sub(r"\s+", " ", text)
    for match in ABBREVIATION_PATTERN.finditer(flattened):
        short_form = match.group(2)
        words = WORD_SEPARATOR.split(match.group(1).strip())
        long_form = " ".join(words[-len(short_form):])
        if initials_match(long_form, short_form):
            found.setdefault(short_form, long_form)
    return found


def spell_out_terms(question: str, terms: dict[str, str]) -> str:
    additions: list[str] = []
    for short_form, long_form in terms.items():
        mentioned = re.search(rf"\b{re.escape(short_form)}\b", question, flags=re.IGNORECASE)
        if mentioned is None or long_form.lower() in question.lower():
            continue
        additions.append(f"{long_form} ({short_form})")
    if additions == []:
        return question
    return f"{question} {' '.join(additions)}"
