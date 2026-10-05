import re
from dataclasses import dataclass

NUMBERED_HEADING = re.compile(r"^\d+(?:\.\d+)*\.?\s+\S")
MAX_PLAIN_HEADING_CHARACTERS = 80
MAX_NUMBERED_HEADING_CHARACTERS = 140


@dataclass(frozen=True)
class Section:
    title: str
    body: str


def plain_heading(line: str) -> bool:
    stripped = line.strip()
    if stripped == "" or len(stripped) > MAX_PLAIN_HEADING_CHARACTERS:
        return False
    return not stripped.endswith((".", "!", "?"))


def numbered_heading(line: str) -> bool:
    stripped = line.strip()
    if stripped == "" or len(stripped) > MAX_NUMBERED_HEADING_CHARACTERS:
        return False
    return NUMBERED_HEADING.match(stripped) is not None


def is_heading(lines: list[str], index: int) -> bool:
    if numbered_heading(lines[index]):
        return True
    previous_blank = index == 0 or lines[index - 1].strip() == ""
    next_blank = index + 1 < len(lines) and lines[index + 1].strip() == ""
    return plain_heading(lines[index]) and previous_blank and (index == 0 or next_blank)


def split_sections(text: str, fallback_title: str) -> list[Section]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    sections: list[Section] = []
    title = fallback_title
    body_lines: list[str] = []
    for index, line in enumerate(lines):
        if not is_heading(lines, index):
            body_lines.append(line)
            continue
        body = "\n".join(body_lines).strip()
        if body != "":
            sections.append(Section(title=title, body=body))
        title = line.strip()
        body_lines = []
    body = "\n".join(body_lines).strip()
    if body != "":
        sections.append(Section(title=title, body=body))
    return sections
