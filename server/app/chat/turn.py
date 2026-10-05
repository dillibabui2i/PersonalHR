import json
import logging
from collections.abc import Callable
from dataclasses import dataclass

from fastapi import HTTPException

from app.activity.service import (
    begin_activity,
    finish_activity,
    note_first_token,
    record_activity_step,
    update_activity_plan,
)
from app.agent.engine import answer_with_engine
from app.activity.trace import TurnTrace
from app.agent.streaming import TurnEvents, step_label
from app.chat.cancel import TurnCancelled
from app.chat.sessions import (
    create_session,
    last_user_question,
    replace_last_no_answer,
    save_message,
    session_is_open,
    session_matches,
    start_new_conversation,
)
from app.documents.service import ready_document_ids
from app.employees.records import (
    ensure_employee,
    ensure_record_facts,
    normalize_email,
    scoped_client_id,
)
from app.organizations.assistant import read_assistant_settings
from app.organizations.service import OrganizationRecord, find_organization_by_host
from app.memory.service import fact_kind, save_fact, stated_facts
from app.rag.generation import generate_general_answer
from app.rag.prompts import NO_ANSWER
from app.rag.types import Citation, RagAnswer, SearchFilter


@dataclass(frozen=True)
class ChatTurn:
    session_id: str
    answer: str
    citations: list[Citation]
    answer_kind: str
    message_id: str = ""
    choices: tuple[str, ...] = ()
    suggestions: tuple[str, ...] = ()
    remembered: bool = False
    created_at: str = ""
    from_page: bool = False


STOPPED_NOTE = "Stopped"
FAILED_NOTE = "The question could not be answered."
turn_log = logging.getLogger("personal_hr")


def active_session_id(requested: str | None, client_id: str, site_host: str, organization_id: str) -> str:
    cleaned = (requested or "").strip()
    if cleaned != "" and session_is_open(cleaned, client_id, site_host):
        return cleaned
    return create_session(client_id, site_host, organization_id)


def traced_answer(trace: TurnTrace, answer: Callable[[], RagAnswer]) -> RagAnswer:
    begin_activity(trace)
    trace.on_step = lambda step: record_activity_step(trace, step)
    trace.on_plan = lambda: update_activity_plan(trace)
    status = "done"
    try:
        return answer()
    except TurnCancelled:
        status = "stopped"
        raise
    except Exception:
        status = "failed"
        raise
    finally:
        if status == "done" and trace.cached:
            status = "cached"
        finish_activity(trace, status)


def attach_events(trace: TurnTrace, events: TurnEvents | None) -> None:
    if events is None:
        return
    trace.cancel = events.cancel

    def on_step_start(name: str) -> None:
        label = step_label(name)
        if label != "":
            events.step(label)

    def on_token(text: str) -> None:
        if text == "":
            return
        trace.streamed_tokens = True
        note_first_token(trace)
        events.token(text)

    trace.on_step_start = on_step_start
    trace.on_token = on_token


def publish_unstreamed(trace: TurnTrace, answer: str) -> None:
    if trace.on_token is None or trace.streamed_tokens or answer in ("", NO_ANSWER):
        return
    trace.on_token(answer)


def answer_from_knowledge(
    organization: OrganizationRecord,
    client_id: str,
    session: str,
    message: str,
    events: TurnEvents | None = None,
    answer_style: str = "brief",
) -> ChatTurn:
    if events is not None:
        events.remember_session(session, "Searching your policies")
    search_filter = SearchFilter(document_ids=ready_document_ids(organization.id))
    trace = TurnTrace(
        session_id=session,
        organization_id=organization.id,
        client_id=client_id,
        question=message,
        route="ANSWER",
        engine="agent",
    )
    attach_events(trace, events)
    user_message_id = save_message(session, "user", message, []).id
    assistant_settings = read_assistant_settings(organization.id)
    page_text = None
    try:
        result = traced_answer(
            trace,
            lambda: answer_with_engine(
                organization.id,
                message,
                search_filter,
                page_text,
                "",
                trace,
                session,
                chosen_answer_style(answer_style),
            ),
        )
    except TurnCancelled:
        stopped = save_message(session, "assistant", STOPPED_NOTE, [], "stopped", run_id=trace.run_id)
        return ChatTurn(
            session_id=session,
            answer=STOPPED_NOTE,
            citations=[],
            answer_kind="stopped",
            message_id=stopped.id,
            created_at=stopped.created_at,
        )
    except Exception:
        turn_log.exception("Chat answer failed")
        return failed_turn(session, trace.run_id)
    if trace.route == "RESET":
        new_session = start_new_conversation(session, client_id, organization.site_host, organization.id)
        if events is not None:
            events.remember_session(new_session, "Starting a new conversation")
        return ChatTurn(session_id=new_session, answer="", citations=[], answer_kind="reset")
    publish_unstreamed(trace, result.answer)
    if trace.route == "CLARIFY" or result.choices != ():
        answer_kind = "clarify"
    elif result.answer == NO_ANSWER:
        answer_kind = "no_answer"
    else:
        answer_kind = "knowledge"
    suggestions = () if answer_kind == "no_answer" else result.suggestions
    saved = save_message(
        session,
        "assistant",
        result.answer,
        result.citations,
        answer_kind,
        result.choices,
        result.from_page,
        suggestions,
        trace.run_id,
    )
    saved_facts = []
    if assistant_settings.memory_enabled:
        planned: list[str] = []
        try:
            payload = json.loads(trace.plan_json or "{}")
            raw_facts = payload.get("statedFacts", [])
            if isinstance(raw_facts, list):
                planned = [item for item in raw_facts if isinstance(item, str)]
        except json.JSONDecodeError:
            planned = []
        saved_facts = [
            save_fact(
                organization.id,
                client_id,
                fact,
                fact_kind(fact),
                user_message_id,
                page_text,
            )
            for fact in stated_facts(message, planned)
        ]
    remembered = any(saved_facts)
    return ChatTurn(
        session_id=session,
        answer=result.answer,
        citations=result.citations,
        answer_kind=answer_kind,
        message_id=saved.id,
        choices=result.choices,
        suggestions=suggestions,
        remembered=remembered,
        created_at=saved.created_at,
        from_page=result.from_page,
    )


def answer_from_general_knowledge(
    organization: OrganizationRecord,
    client_id: str,
    session_id: str | None,
    events: TurnEvents | None = None,
) -> ChatTurn:
    session = (session_id or "").strip()
    if session == "" or not session_matches(session, client_id, organization.site_host):
        raise HTTPException(status_code=404, detail="That conversation was not found.")
    question = last_user_question(session)
    if question == "":
        raise HTTPException(status_code=400, detail="Ask a question first.")
    if events is not None:
        events.remember_session(session, "Answering from general knowledge")
    trace = TurnTrace(
        session_id=session,
        organization_id=organization.id,
        client_id=client_id,
        question=question,
        route="general",
        engine="agent",
    )
    attach_events(trace, events)
    try:
        result = traced_answer(trace, lambda: generate_general_answer(question, trace))
    except TurnCancelled:
        stopped = save_message(session, "assistant", STOPPED_NOTE, [], "stopped", run_id=trace.run_id)
        return ChatTurn(
            session_id=session,
            answer=STOPPED_NOTE,
            citations=[],
            answer_kind="stopped",
            message_id=stopped.id,
            created_at=stopped.created_at,
        )
    except Exception:
        turn_log.exception("General answer failed")
        return failed_turn(session, trace.run_id)
    publish_unstreamed(trace, result.answer)
    saved = replace_last_no_answer(session, result.answer, "general", trace.run_id)
    if saved is None:
        saved = save_message(session, "assistant", result.answer, [], "general", run_id=trace.run_id)
    return ChatTurn(
        session_id=session,
        answer=result.answer,
        citations=[],
        answer_kind="general",
        message_id=saved.id,
        created_at=saved.created_at,
    )


def failed_turn(session_id: str, run_id: str) -> ChatTurn:
    saved = save_message(session_id, "assistant", FAILED_NOTE, [], "failed", run_id=run_id)
    return ChatTurn(
        session_id=session_id,
        answer=FAILED_NOTE,
        citations=[],
        answer_kind="failed",
        message_id=saved.id,
        created_at=saved.created_at,
    )


def chosen_answer_style(value: str) -> str:
    return "detailed" if value.strip().lower() == "detailed" else "brief"


def take_chat_turn(
    client_id: str,
    site_host: str,
    message: str,
    session_id: str | None,
    answer_generally: bool = False,
    events: TurnEvents | None = None,
    employee_email: str = "",
    answer_style: str = "brief",
) -> ChatTurn:
    cleaned_client = client_id.strip()
    if cleaned_client == "":
        raise HTTPException(status_code=400, detail="Enter a question.")
    organization = find_organization_by_host(site_host)
    scope = scoped_client_id(cleaned_client, site_host, employee_email)
    email = normalize_email(employee_email)
    if email != "":
        employee = ensure_employee(organization.id, email)
        if employee.onboarded_at == "":
            raise HTTPException(status_code=409, detail="Finish onboarding before asking a question.")
        ensure_record_facts(employee)
    if answer_generally:
        return answer_from_general_knowledge(organization, scope, session_id, events)
    cleaned_message = message.strip()
    if cleaned_message == "":
        raise HTTPException(status_code=400, detail="Enter a question.")
    session = active_session_id(session_id, scope, organization.site_host, organization.id)
    return answer_from_knowledge(
        organization,
        scope,
        session,
        cleaned_message,
        events,
        chosen_answer_style(answer_style),
    )
