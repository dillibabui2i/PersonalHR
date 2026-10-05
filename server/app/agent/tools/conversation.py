import logging
import re
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from app.activity.trace import open_step
from app.agent.context import active_turn
from app.chat.sessions import list_messages

agent_log = logging.getLogger("personal_hr.agents")

WORD = re.compile(r"[a-z0-9']+")
STOP_WORDS = {
    "a",
    "about",
    "an",
    "and",
    "answer",
    "ask",
    "asked",
    "chat",
    "conversation",
    "did",
    "discuss",
    "discussed",
    "do",
    "earlier",
    "for",
    "i",
    "in",
    "it",
    "last",
    "me",
    "mentioned",
    "message",
    "my",
    "of",
    "on",
    "or",
    "please",
    "previous",
    "remind",
    "say",
    "said",
    "tell",
    "that",
    "the",
    "this",
    "those",
    "time",
    "to",
    "told",
    "was",
    "we",
    "were",
    "what",
    "you",
}
MATCH_LIMIT = 6


class ConversationSearchInput(BaseModel):
    query: str = Field(description="Words to find in earlier messages of this conversation.")


def search_terms(query: str) -> list[str]:
    return [word for word in WORD.findall(query.lower()) if word not in STOP_WORDS and len(word) > 1]


def overlap(text: str, terms: list[str]) -> int:
    lowered = text.lower()
    score = sum(1 for term in terms if re.search(rf"\b{re.escape(term)}\b", lowered) is not None)
    if len(terms) >= 2 and " ".join(terms) in lowered:
        score += len(terms)
    return score


def matching_messages(
    messages: list[tuple[str, str]],
    query: str,
    limit: int = MATCH_LIMIT,
) -> list[tuple[str, str]]:
    """Return earlier turns that mention the query, or the latest exchange when the query has no topic."""
    if messages == []:
        return []
    terms = search_terms(query)
    if terms == []:
        for index in range(len(messages) - 1, -1, -1):
            if messages[index][0] == "user":
                return messages[index:]
        return messages[-2:]
    scored = [
        (index, overlap(content, terms))
        for index, (_role, content) in enumerate(messages)
        if overlap(content, terms) > 0
    ]
    if scored == []:
        return []
    scored.sort(key=lambda item: (item[1], item[0]), reverse=True)
    chosen = {index for index, _score in scored[:limit]}
    for index in list(chosen):
        if messages[index][0] != "assistant":
            continue
        for earlier in range(index - 1, -1, -1):
            chosen.add(earlier)
            if messages[earlier][0] == "user":
                break
    return [messages[index] for index in sorted(chosen)]


def format_matches(messages: list[tuple[str, str]]) -> str:
    lines: list[str] = []
    for role, content in messages:
        speaker = "Employee" if role == "user" else "Assistant"
        shortened = " ".join(content.split())
        if len(shortened) > 500:
            shortened = f"{shortened[:499].rstrip()}…"
        lines.append(f"{speaker}: {shortened}")
    return "\n".join(lines)


class SearchConversationTool(BaseTool):
    name: str = "search_conversation"
    description: str = (
        "Search earlier messages in this conversation. "
        "Use it when the employee asks what was said, asked, or decided earlier in the chat."
    )
    args_schema: Type[BaseModel] = ConversationSearchInput

    def _run(self, query: str) -> str:
        return self.execute(query)

    def execute(self, query: str) -> str:
        context = active_turn.get()
        if context is None or context.session_id.strip() == "":
            return ""
        agent_log.info("flow is calling search_conversation")
        messages = list_messages(context.session_id)
        if messages != [] and messages[-1].role == "user":
            messages = messages[:-1]
        turns = [(message.role, message.content) for message in messages]
        with open_step(context.trace, "search_conversation", query.strip(), "tool") as outcome:
            matched = matching_messages(turns, query)
            text = format_matches(matched)
            outcome.output_summary = "No earlier messages matched." if text == "" else f"{len(matched)} messages"
            if text == "":
                agent_log.info("search_conversation returned no messages to flow")
            else:
                agent_log.info("search_conversation returned %d messages to flow", len(matched))
            return text
