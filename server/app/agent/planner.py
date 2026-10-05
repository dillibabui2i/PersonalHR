import re
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from app.agent.clarify import (
    document_leave_context,
    hinted_leave,
    leave_for_code,
    mentioned_leave,
    replace_leave_code,
)
from app.core.settings import load_settings
from app.llm.chat_model import ChatModelFailed, generate_json
from app.memory.service import facts_context, recall_facts, stated_facts
from app.rag.date_math import date_pair

RouteName = Literal["GREETING", "OUT_OF_SCOPE", "ANSWER", "CLARIFY", "RESET"]

GREETING_REPLY = (
    "Hey {name}, welcome back! I can help with your organization’s policies and leave."
)

PLANNER_PROMPT = """Plan one turn for a private HR assistant.

The assistant may answer only from the organization's uploaded documents, date arithmetic,
or facts the employee has shared. It does not read portal pages. A personal leave balance
is only a number the employee states.

Return:
- route GREETING for a greeting, thanks, or a question about who the assistant is.
- route OUT_OF_SCOPE when the request is unrelated to those allowed sources.
- route ANSWER otherwise.
- needsPolicy when uploaded documents are needed for a rule, allowance, leave type, or definition.
- needsPortal is always false.
- needsMemory when the employee asks about a fact they previously shared.
- needsConversation when the employee asks what was said, asked, or decided earlier in this chat.
  Leave it false for a new policy question.
- needsDates when the question names a date span.
- startDate and endDate as YYYY-MM-DD when needsDates is true, otherwise empty strings.
- standaloneQuestion as one question that can be understood without the conversation.
  Resolve "that", "it", "those days", and phrases such as "and for sick?" from the recent turns.
  If the latest message changes the subject, keep the new subject and do not mention the old one.
  Write a clear policy search question: keep the employee's intent, stated leave balance,
  and topic words such as encashment. Do not copy typos or extra requests to "elaborate".
  Keep a balance the employee states, such as "I have 18 days", in standaloneQuestion.
  A question about when they will reach a threshold stays on that subject and still needs the policy.
  Do not turn that question into a request to pick a leave type.
- statedFacts updates durable core memory when this turn contains a personal fact worth keeping.
  The model decides what matters: name, role, team, manager, office, joining date, or similar
  self-stated details, even without the word "remember". Prefer the employee's own wording.
  If a new fact replaces an older one of the same kind, put only the new fact in statedFacts.
  Leave statedFacts empty for ordinary policy questions with no personal fact.
  Never include passwords, passcodes, tokens, secrets, leave balances, or day counts.
A personal balance is never answered from a policy allowance. A date span still needs the policy when the question asks whether a rule allows it.

Employee-stated facts:
{facts}

Recent conversation:
{history}

Latest message:
{message}
"""

FOLLOW_UP = re.compile(r"^\s*(and\b|what about\b|how about\b)|\b(that|it|those|this)\b", re.IGNORECASE)
WHEN_THRESHOLD = re.compile(
    r"\bwhen\b.{0,50}\b(?:get|reach|earn|credit|eligible|eligib)|"
    r"\b(?:this|that|those)\b.{0,20}\b\d+(?:\.\d+)?\s+days?\b",
    re.IGNORECASE,
)
AMBIGUOUS_LEAVE = re.compile(r"\bhow many\b.{0,60}\bleaves?\b", re.IGNORECASE)
NOTE_ASK = re.compile(r"\bnote\b", re.IGNORECASE)
COUNT_ASK = re.compile(r"\b(how many|can i take|days)\b", re.IGNORECASE)


class TurnPlan(BaseModel):
    standalone_question: str = Field(alias="standaloneQuestion")
    route: RouteName
    needs_policy: bool = Field(alias="needsPolicy")
    needs_portal: bool = Field(alias="needsPortal")
    needs_memory: bool = Field(default=False, alias="needsMemory")
    needs_conversation: bool = Field(default=False, alias="needsConversation")
    needs_dates: bool = Field(default=False, alias="needsDates")
    start_date: str = Field(default="", alias="startDate")
    end_date: str = Field(default="", alias="endDate")
    clarifying_question: str = Field(default="", alias="clarifyingQuestion")
    choices: list[str] = Field(default_factory=list)
    stated_facts: list[str] = Field(default_factory=list, alias="statedFacts")

    model_config = {"populate_by_name": True}


PERSONAL_PAGE = re.compile(
    r"\b(balance|left|remaining|timesheet|leave details|leave request)\b|\bhow many\b.{0,50}\bhave\b",
    re.IGNORECASE,
)
MEMORY_QUERY = re.compile(
    r"\b(when did i join|how long have i worked|where do i work|which office|what do you remember)\b",
    re.IGNORECASE,
)
CONVERSATION_QUERY = re.compile(
    r"\b("
    r"what did (?:you|i|we) (?:say|ask|tell|talk|discuss)"
    r"|you (?:said|mentioned|told)"
    r"|(?:i|we) (?:said|asked|told|talked|discussed)"
    r"|earlier|last time|remind me"
    r"|in this (?:chat|conversation)"
    r"|previous (?:answer|message|question)"
    r")\b",
    re.IGNORECASE,
)
MEMORY_COMMAND = re.compile(r"\bremember\b|\bI\s+(?:joined|started work|work from|am based)\b", re.IGNORECASE)
LEAVE_REQUEST = re.compile(r"\b(i want|can i|notice|allow)\b", re.IGNORECASE)
POLICY_TOPIC = re.compile(
    r"\b(leave|carried? forward|notice|medical note|allowance|eligib(?:le|ility)|encash|credit|accrual|quarter)\b",
    re.IGNORECASE,
)
EXPAND_PREVIOUS = re.compile(
    r"\b(elaborat\w*|explain(?:\s+more)?|more details?|examples?|in detail)\b",
    re.IGNORECASE,
)
ALLOWANCE_QUESTION = re.compile(
    r"\bhow many\b.{0,60}\b(can i take|do i get|am i allowed|days can i)\b",
    re.IGNORECASE,
)
TAKE_LEAVE = re.compile(r"\b(i want|i need|i'd like|apply|book|take|taking)\b", re.IGNORECASE)
DURATION = re.compile(r"\b\d+(?:\.\d+)?\s+days?\b", re.IGNORECASE)
STATED_BALANCE = re.compile(
    r"\b(?:i have|i've got|i currently have|my balance is)\b.{0,40}\b\d+(?:\.\d+)?\b",
    re.IGNORECASE,
)
STATED_BALANCE_NUMBER = re.compile(
    r"\b(?:i have|i've got|i currently have|my balance is)\b.{0,40}?(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
ASKED_BALANCE = re.compile(r"how many (.+?) days do you have\??", re.IGNORECASE)
ANSWER_REQUEST = re.compile(
    r"\b(i need|i want|i'd like)\b.{0,50}\b(answer|response|example|detail|explanation|elaborat\w*)\b",
    re.IGNORECASE,
)


def deterministic_plan(message: str) -> TurnPlan | None:
    normalized = re.sub(r"[.!?]+$", "", " ".join(message.lower().split())).strip()
    if normalized in {"start over", "new conversation", "reset"}:
        return TurnPlan(
            standaloneQuestion=message.strip(),
            route="RESET",
            needsPolicy=False,
            needsPortal=False,
        )
    greetings = {
        "hello",
        "hi",
        "hey",
        "thanks",
        "thank you",
        "who are you",
        "what can you do",
    }
    if normalized not in greetings:
        facts = stated_facts(message)
        if MEMORY_QUERY.search(message) is not None:
            return TurnPlan(
                standaloneQuestion=message.strip(),
                route="ANSWER",
                needsPolicy=False,
                needsPortal=False,
                needsMemory=True,
            )
        if MEMORY_COMMAND.search(message) is None:
            return None
        return TurnPlan(
            standaloneQuestion=message.strip(),
            route="ANSWER",
            needsPolicy=False,
            needsPortal=False,
            needsMemory=False,
            statedFacts=facts,
        )
    return TurnPlan(
        standaloneQuestion=message.strip(),
        route="GREETING",
        needsPolicy=False,
        needsPortal=False,
    )


def fallback_plan(message: str) -> TurnPlan:
    return TurnPlan(
        standaloneQuestion=message.strip(),
        route="ANSWER",
        needsPolicy=True,
        needsPortal=False,
    )


def cleaned_text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def normalize_plan(raw: dict[str, object]) -> dict[str, object]:
    normalized = dict(raw)
    normalized["needsDates"] = normalized.get("needsDates") is True
    normalized["needsMemory"] = normalized.get("needsMemory") is True
    normalized["needsConversation"] = normalized.get("needsConversation") is True
    normalized["startDate"] = cleaned_text(normalized.get("startDate"))
    normalized["endDate"] = cleaned_text(normalized.get("endDate"))
    return normalized


def apply_question_signals(message: str, plan: TurnPlan, today: date | None = None) -> TurnPlan:
    pair = date_pair(message, date.today() if today is None else today)
    leave_request = LEAVE_REQUEST.search(message) is not None
    memory_query = MEMORY_QUERY.search(message) is not None
    policy_topic = POLICY_TOPIC.search(message) is not None
    conversation_query = CONVERSATION_QUERY.search(message) is not None
    if pair is not None:
        plan.needs_dates = True
        plan.start_date = pair[0].isoformat()
        plan.end_date = pair[1].isoformat()
        plan.route = "ANSWER"
    if pair is not None and leave_request:
        plan.needs_policy = True
        plan.route = "ANSWER"
    if memory_query:
        plan.needs_memory = True
        plan.needs_policy = False
        plan.needs_portal = False
        plan.route = "ANSWER"
    elif policy_topic or WHEN_THRESHOLD.search(message) is not None:
        plan.needs_policy = True
        plan.route = "ANSWER"
    if conversation_query:
        plan.needs_conversation = True
        plan.route = "ANSWER"
        if not policy_topic and not memory_query and pair is None:
            plan.needs_policy = False
            plan.needs_portal = False
    if plan.needs_dates and (plan.start_date == "" or plan.end_date == ""):
        plan.needs_dates = False
    plan.needs_portal = False
    if (
        plan.route == "ANSWER"
        and not plan.needs_policy
        and not plan.needs_portal
        and not plan.needs_dates
        and not plan.needs_memory
        and not plan.needs_conversation
        and plan.stated_facts == []
        and MEMORY_COMMAND.search(message) is None
    ):
        plan.needs_policy = True
    return plan


def keep_conversation_lookup(message: str, plan: TurnPlan) -> TurnPlan:
    """Keep a question about earlier messages even when the planner rewrites it."""
    if CONVERSATION_QUERY.search(message) is None or plan.route not in {"ANSWER", "OUT_OF_SCOPE"}:
        return plan
    plan.needs_conversation = True
    plan.route = "ANSWER"
    mentions_policy = POLICY_TOPIC.search(message) is not None or POLICY_TOPIC.search(plan.standalone_question) is not None
    if not mentions_policy and MEMORY_QUERY.search(message) is None and not plan.needs_dates:
        plan.needs_policy = False
        plan.needs_portal = False
    return plan


def plan_payload(plan: TurnPlan) -> dict[str, object]:
    dates = None
    if plan.start_date != "" and plan.end_date != "":
        dates = {"startDate": plan.start_date, "endDate": plan.end_date}
    return {
        "standaloneQuestion": plan.standalone_question,
        "route": plan.route,
        "needsPolicy": plan.needs_policy,
        "needsPortal": plan.needs_portal,
        "needsMemory": plan.needs_memory,
        "needsConversation": plan.needs_conversation,
        "needsDates": plan.needs_dates,
        "dates": dates,
        "clarifyingQuestion": plan.clarifying_question,
        "choices": plan.choices,
        "statedFacts": plan.stated_facts,
    }


def last_user_text(history: list[tuple[str, str]]) -> str:
    for role, content in reversed(history):
        if role == "user":
            return content
    return ""


def last_exchange(history: list[tuple[str, str]]) -> list[tuple[str, str]]:
    for index in range(len(history) - 1, -1, -1):
        if history[index][0] == "user":
            return history[index:]
    return []


def recent_topic(history: list[tuple[str, str]], labels: list[str] | None = None) -> str:
    """Leave type the employee named in the previous question, not in an older reply."""
    for role, content in last_exchange(history):
        if role == "user":
            return mentioned_leave(content, labels)
    return ""


def latest_topic(history: list[tuple[str, str]], labels: list[str] | None = None) -> str:
    for _role, content in reversed(history):
        topic = mentioned_leave(content, labels)
        if topic != "":
            return topic
    return ""


def format_history(history: list[tuple[str, str]]) -> str:
    if history == []:
        return "none"
    lines: list[str] = []
    for role, content in history:
        speaker = "Employee" if role == "user" else "Assistant"
        shortened = " ".join(content.split())
        if len(shortened) > 300:
            shortened = f"{shortened[:299].rstrip()}…"
        lines.append(f"{speaker}: {shortened}")
    return "\n".join(lines)


def rewrite_follow_up(
    message: str,
    history: list[tuple[str, str]],
    labels: list[str] | None = None,
) -> str | None:
    """Turn a short follow-up into a question that names the subject from earlier turns."""
    if history == [] or FOLLOW_UP.search(message) is None:
        return None
    known = [] if labels is None else labels
    explicit = hinted_leave(message, known)
    if explicit == "" and message.lower().lstrip().startswith("and "):
        return None
    topic = explicit if explicit != "" else recent_topic(history, known)
    if topic == "":
        if WHEN_THRESHOLD.search(message) is None and DURATION.search(message) is None:
            return None
        prior = last_user_text(history).strip()
        if prior == "":
            return None
        return f"{message.strip()} {prior}"
    if explicit != "" and COUNT_ASK.search(last_user_text(history)) is not None:
        return f"How many {topic} days can I take?"
    if explicit != "":
        return f"What is the rule for {topic}?"
    if NOTE_ASK.search(message) is not None:
        return f"Do I need a medical note for {topic}?"
    if re.search(r"\bthose days\b", message, re.IGNORECASE) is not None:
        return f"What is the rule for those {topic} days?"
    return f"What is the rule for {topic}?"


def unresolved_follow_up(
    message: str,
    history: list[tuple[str, str]],
    labels: list[str] | None = None,
) -> bool:
    """A short follow-up that names no subject and has no earlier turn to borrow one from."""
    if FOLLOW_UP.search(message) is None or mentioned_leave(message, labels) != "":
        return False
    if WHEN_THRESHOLD.search(message) is not None or DURATION.search(message) is not None:
        return False
    if history == []:
        return True
    return recent_topic(history, labels) == "" and latest_topic(history, labels) == ""


def ambiguous_leave_question(message: str, labels: list[str] | None = None) -> bool:
    """True only for an unnamed allowance question, such as 'how many leaves can I take?'."""
    if mentioned_leave(message, labels) != "" or EXPAND_PREVIOUS.search(message) is not None:
        return False
    return AMBIGUOUS_LEAVE.search(message) is not None


def last_assistant_text(history: list[tuple[str, str]]) -> str:
    for role, content in reversed(history):
        if role == "assistant":
            return content
    return ""


def asks_to_use_leave(message: str) -> bool:
    """True when the employee is booking leave, not asking for an explanation."""
    if TAKE_LEAVE.search(message) is None:
        return False
    if ANSWER_REQUEST.search(message) is None:
        return True
    return re.search(r"\b(apply|book|take|taking)\b", message, re.IGNORECASE) is not None


def needs_their_balance(message: str) -> bool:
    """True when the employee wants to use leave, or asks for their own balance."""
    if message.strip() == "" or ALLOWANCE_QUESTION.search(message) is not None:
        return False
    if STATED_BALANCE.search(message) is not None or EXPAND_PREVIOUS.search(message) is not None:
        return False
    if PERSONAL_PAGE.search(message) is not None and re.search(r"\b(balance|left|remaining|have)\b", message, re.IGNORECASE):
        return True
    return asks_to_use_leave(message) and (
        DURATION.search(message) is not None or re.search(r"\bleaves?\b", message, re.IGNORECASE) is not None
    )


def balance_already_known(message: str, history: list[tuple[str, str]]) -> bool:
    if STATED_BALANCE.search(message) is not None:
        return True
    return any(role == "user" and STATED_BALANCE.search(content) is not None for role, content in history)


def leave_request_text(history: list[tuple[str, str]]) -> str:
    for role, content in reversed(history):
        if role == "user" and needs_their_balance(content):
            return content
    return last_user_text(history)


def continue_after_balance(message: str, history: list[tuple[str, str]]) -> TurnPlan | None:
    """Use a number the employee just gave as their balance for the leave they asked about."""
    asked = ASKED_BALANCE.search(last_assistant_text(history))
    number = re.search(r"\b(\d+(?:\.\d+)?)\b", message)
    if asked is None or number is None:
        return None
    topic = " ".join(asked.group(1).split())
    prior = leave_request_text(history).strip()
    balance = f"I have {number.group(1)} {topic} days."
    standalone = balance if prior == "" else f"{prior.rstrip('.')}. {balance}"
    plan = TurnPlan(
        standaloneQuestion=standalone,
        route="ANSWER",
        needsPolicy=True,
        needsPortal=False,
    )
    pair = date_pair(prior, date.today())
    if pair is not None:
        plan.needs_dates = True
        plan.start_date = pair[0].isoformat()
        plan.end_date = pair[1].isoformat()
    return plan


def balance_plan(
    message: str,
    history: list[tuple[str, str]],
    organization_id: str,
    labels: list[str] | None = None,
    choices: list[str] | None = None,
) -> TurnPlan | None:
    """Ask for the employee's own balance. Leave types come from this organization's policy."""
    if not needs_their_balance(message) or balance_already_known(message, history):
        return None
    if labels is None or choices is None:
        labels, choices = document_leave_context(organization_id)
    topic = mentioned_leave(message, labels) or recent_topic(history, labels)
    if topic == "":
        if len(choices) >= 2:
            return TurnPlan(
                standaloneQuestion=message.strip(),
                route="CLARIFY",
                needsPolicy=False,
                needsPortal=False,
                clarifyingQuestion="Which kind of leave do you mean?",
                choices=choices,
            )
        topic = labels[0] if labels else "leave"
    return TurnPlan(
        standaloneQuestion=message.strip(),
        route="CLARIFY",
        needsPolicy=False,
        needsPortal=False,
        clarifyingQuestion=f"How many {topic} days do you have?",
    )


def rewrite_choice(
    message: str,
    history: list[tuple[str, str]],
    labels: list[str] | None = None,
) -> str | None:
    """A button label that names one leave type continues the question that asked which kind."""
    topic = mentioned_leave(message, labels) or mentioned_leave(message)
    normalized = " ".join(message.lower().split()).rstrip(".!?")
    if topic == "" or normalized not in {topic, f"{topic}s"}:
        return None
    previous = last_user_text(history)
    if needs_their_balance(previous):
        return f"{previous.rstrip('.')}. The leave type is {topic}."
    if not ambiguous_leave_question(previous, labels):
        return None
    return f"How many {topic} days can I take?"


def apply_clarification(
    message: str,
    history: list[tuple[str, str]],
    plan: TurnPlan,
    organization_id: str,
    page_text: str = "",
    labels: list[str] | None = None,
    choices: list[str] | None = None,
) -> TurnPlan:
    if plan.route == "CLARIFY":
        plan.route = "ANSWER"
    if plan.route != "ANSWER" or not ambiguous_leave_question(message, labels):
        return plan
    topic = recent_topic(history, labels)
    if topic != "":
        plan.standalone_question = f"How many {topic} days can I take?"
        plan.needs_policy = True
        plan.route = "ANSWER"
        return plan
    if labels is None or choices is None:
        labels, choices = known_leave_labels(organization_id, page_text)
    if len(choices) < 2:
        return plan
    plan.route = "CLARIFY"
    plan.clarifying_question = "Which kind of leave do you mean?"
    plan.choices = choices
    plan.needs_policy = False
    plan.needs_portal = False
    plan.needs_dates = False
    plan.standalone_question = message.strip()
    return plan


def keep_employee_question(
    original: str,
    standalone: str,
    history: list[tuple[str, str]] | None = None,
    labels: list[str] | None = None,
    trust_leave: bool = False,
) -> str:
    """Keep a stated balance. A leave type stays when this turn or an earlier turn named it."""
    cleaned = standalone.strip() or original.strip()
    turns = [] if history is None else history
    introduced = mentioned_leave(cleaned, labels)
    carried = introduced != "" and introduced in {recent_topic(turns, labels), latest_topic(turns, labels)}
    if not trust_leave and introduced != "" and mentioned_leave(original, labels) == "" and not carried:
        cleaned = original.strip()
    if STATED_BALANCE.search(original) is not None and STATED_BALANCE.search(cleaned) is None:
        number = STATED_BALANCE_NUMBER.search(original)
        if number is not None:
            cleaned = f"{cleaned.rstrip('.')} I have {number.group(1)} days."
    return cleaned


def preserve_topic_change(
    message: str,
    standalone: str,
    history: list[tuple[str, str]],
    labels: list[str] | None = None,
) -> str:
    cleaned = standalone.strip() or message.strip()
    if FOLLOW_UP.search(message) is not None:
        return cleaned
    old_topic = recent_topic(history, labels)
    new_topic = mentioned_leave(cleaned, labels)
    if old_topic != "" and new_topic == old_topic and mentioned_leave(message, labels) == "":
        return message.strip()
    return cleaned


def known_leave_labels(organization_id: str, page_text: str = "") -> tuple[list[str], list[str]]:
    del page_text
    return document_leave_context(organization_id)


def leave_context(
    message: str,
    history: list[tuple[str, str]],
    organization_id: str,
    page_text: str,
) -> tuple[list[str], list[str]] | None:
    """Leave names from this organization's documents when the turn is about leave or a follow-up."""
    parts = [message]
    for role, content in history:
        if role == "user":
            parts.append(content)
    blob = "\n".join(parts)
    about_leave = re.search(r"\bleaves?\b", blob, re.IGNORECASE) is not None
    if not about_leave and FOLLOW_UP.search(message) is None:
        return None
    return known_leave_labels(organization_id, page_text)


def question_with_leave_name(message: str, labels: list[str]) -> str:
    """Expand a short form from this organization's documents, such as PL for Privilege leave."""
    if mentioned_leave(message, labels) != "":
        return message
    resolved = leave_for_code(message, labels)
    if resolved == "":
        return message
    return replace_leave_code(message, resolved)


def understand_turn(
    message: str,
    history: list[tuple[str, str]] | None = None,
    organization_id: str = "",
    client_id: str = "",
    page_text: str = "",
    use_memory: bool = True,
) -> TurnPlan:
    turns = [] if history is None else history
    original = message
    direct = deterministic_plan(original)
    if direct is not None:
        return direct
    continued = continue_after_balance(original, turns)
    if continued is not None:
        return continued
    loaded = leave_context(original, turns, organization_id, page_text)
    labels, choices = ([], []) if loaded is None else loaded
    known = None if loaded is None else labels
    picked = None if loaded is None else choices
    message = question_with_leave_name(original, labels)
    if unresolved_follow_up(message, turns, known):
        return TurnPlan(
            standaloneQuestion=message.strip(),
            route="CLARIFY",
            needsPolicy=False,
            needsPortal=False,
            clarifyingQuestion="What are you asking about?",
        )
    rewritten = rewrite_choice(message, turns, known) or rewrite_follow_up(message, turns, labels)
    if rewritten is None and ambiguous_leave_question(message, known):
        topic = recent_topic(turns, known)
        if topic != "":
            rewritten = f"How many {topic} days can I take?"
        elif len(choices) >= 2:
            return TurnPlan(
                standaloneQuestion=message.strip(),
                route="CLARIFY",
                needsPolicy=False,
                needsPortal=False,
                clarifyingQuestion="Which kind of leave do you mean?",
                choices=choices,
            )
    planned = rewritten if rewritten is not None else message.strip()
    if needs_their_balance(planned) and not balance_already_known(planned, turns):
        asked = balance_plan(planned, turns, organization_id, known, picked)
        if asked is not None:
            return asked
    settings = load_settings()
    remembered = ""
    if use_memory:
        remembered = facts_context(recall_facts(organization_id, client_id, settings.employee_fact_limit))
    try:
        result = generate_json(
            PLANNER_PROMPT.format(
                facts=remembered if remembered != "" else "none",
                history=format_history(turns),
                message=planned,
            ),
            TurnPlan.model_json_schema(by_alias=True),
            45,
            settings.planner_model,
            256,
        )
        plan = TurnPlan.model_validate(normalize_plan(result))
        if plan.route == "RESET":
            plan.route = "ANSWER"
    except (ChatModelFailed, ValueError):
        plan = fallback_plan(planned)
    plan.stated_facts = stated_facts(original, plan.stated_facts)
    if rewritten is not None:
        plan.standalone_question = keep_employee_question(original, rewritten, turns, known, trust_leave=True)
        plan.route = "ANSWER"
    else:
        plan.standalone_question = keep_employee_question(
            original,
            preserve_topic_change(message, plan.standalone_question, turns, known),
            turns,
            known,
        )
    plan = apply_question_signals(plan.standalone_question, plan)
    if rewritten is None:
        plan = apply_clarification(message, turns, plan, organization_id, page_text, known, picked)
    plan = keep_conversation_lookup(original, plan)
    if EXPAND_PREVIOUS.search(original) is not None and turns != []:
        plan.route = "ANSWER"
        plan.needs_conversation = True
        plan.clarifying_question = ""
        plan.choices = []
        if POLICY_TOPIC.search(original) is not None or POLICY_TOPIC.search(plan.standalone_question) is not None:
            plan.needs_policy = True
    if WHEN_THRESHOLD.search(original) is not None:
        plan.route = "ANSWER"
        plan.needs_policy = True
        plan.needs_conversation = True
        plan.needs_memory = True
        plan.clarifying_question = ""
        plan.choices = []
    if plan.route != "ANSWER":
        return plan
    if not needs_their_balance(original) or balance_already_known(original, turns):
        return plan
    asked = balance_plan(plan.standalone_question, turns, organization_id, known, picked)
    return asked if asked is not None else plan
