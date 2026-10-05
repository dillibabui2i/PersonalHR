import asyncio
import logging
import queue
import threading

from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from app.admin.session import require_organization_access
from app.agent.streaming import SSE_HEADERS, TurnEvents, encode_event, register_turn, release_turn, stop_turn
from app.chat.feedback import save_answer_feedback
from app.chat.sessions import decline_last_no_answer, list_messages, session_matches, start_new_conversation
from app.chat.turn import ChatTurn, take_chat_turn
from app.core.settings import load_settings
from app.employees.records import scoped_client_id
from app.memory.service import forget_conversation_facts, forget_fact, recall_facts
from app.organizations.service import find_organization_by_host, normalize_site_host
from app.rag.types import Citation

request_log = logging.getLogger("personal_hr")

router = APIRouter()


def require_site_access(token: str, site_host: str) -> None:
    organization = find_organization_by_host(site_host)
    require_organization_access(token, organization.id)


class ChatRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    client_id: str = Field(default="", validation_alias="clientId")
    session_id: str | None = Field(default=None, validation_alias="sessionId")
    site_host: str = Field(default="", validation_alias="siteHost")
    message: str = ""
    answer_generally: bool = Field(default=False, validation_alias="answerGenerally")
    answer_style: str = Field(default="brief", validation_alias="answerStyle")
    employee_email: str = Field(default="", validation_alias="employeeEmail")


class CitationResponse(BaseModel):
    document_name: str = Field(serialization_alias="documentName")
    section_title: str = Field(serialization_alias="sectionTitle")
    excerpt: str = ""


class ChatResponse(BaseModel):
    session_id: str = Field(serialization_alias="sessionId")
    message_id: str = Field(serialization_alias="messageId")
    answer: str
    citations: list[CitationResponse]
    answer_kind: str = Field(serialization_alias="answerKind")
    choices: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    remembered: bool = False
    created_at: str = Field(default="", serialization_alias="createdAt")
    from_page: bool = Field(default=False, serialization_alias="fromPage")


def citation_response(citation: Citation) -> CitationResponse:
    return CitationResponse(
        document_name=citation.document_name,
        section_title=citation.section_title,
        excerpt=citation.excerpt,
    )


def chat_response(turn: ChatTurn) -> ChatResponse:
    return ChatResponse(
        session_id=turn.session_id,
        message_id=turn.message_id,
        answer=turn.answer,
        citations=[citation_response(citation) for citation in turn.citations],
        answer_kind=turn.answer_kind,
        choices=list(turn.choices),
        suggestions=list(turn.suggestions),
        remembered=turn.remembered,
        created_at=turn.created_at,
        from_page=turn.from_page,
    )


class HistoryMessage(BaseModel):
    id: str
    role: str
    content: str
    citations: list[CitationResponse]
    answer_kind: str = Field(serialization_alias="answerKind")
    choices: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    remembered: bool = False
    created_at: str = Field(serialization_alias="createdAt")
    from_page: bool = Field(default=False, serialization_alias="fromPage")
    rating: str = ""
    feedback_note: str = Field(default="", serialization_alias="feedbackNote")


class ResetResponse(BaseModel):
    session_id: str = Field(serialization_alias="sessionId")


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    client_id: str = Field(default="", validation_alias="clientId")
    session_id: str = Field(default="", validation_alias="sessionId")
    site_host: str = Field(default="", validation_alias="siteHost")
    message_id: str = Field(default="", validation_alias="messageId")
    rating: str = ""
    note: str = ""
    employee_email: str = Field(default="", validation_alias="employeeEmail")


class FeedbackResponse(BaseModel):
    rating: str
    note: str


def rate_answer(body: FeedbackRequest) -> FeedbackResponse:
    rating, note = save_answer_feedback(
        body.message_id,
        body.session_id,
        scoped_client_id(body.client_id, body.site_host, body.employee_email),
        body.site_host,
        body.rating,
        body.note,
    )
    return FeedbackResponse(rating=rating, note=note)


class MemoryFactResponse(BaseModel):
    id: str
    fact: str
    kind: str
    created_at: str = Field(serialization_alias="createdAt")


def decline_chat(body: ChatRequest) -> dict[str, bool]:
    cleaned_client = scoped_client_id(body.client_id, body.site_host, body.employee_email)
    session_id = (body.session_id or "").strip()
    if cleaned_client == "" or session_id == "":
        raise HTTPException(status_code=400, detail="That conversation was not found.")
    if not session_matches(session_id, cleaned_client, normalize_site_host(body.site_host)):
        raise HTTPException(status_code=404, detail="That conversation was not found.")
    if not decline_last_no_answer(session_id):
        raise HTTPException(status_code=409, detail="That offer is no longer available.")
    return {"declined": True}


def reset_chat(body: ChatRequest) -> ResetResponse:
    cleaned_client = scoped_client_id(body.client_id, body.site_host, body.employee_email)
    if cleaned_client == "":
        raise HTTPException(status_code=400, detail="Enter a question.")
    organization = find_organization_by_host(body.site_host)
    session_id = start_new_conversation(
        body.session_id,
        cleaned_client,
        organization.site_host,
        organization.id,
    )
    return ResetResponse(session_id=session_id)


class HistoryResponse(BaseModel):
    session_id: str = Field(serialization_alias="sessionId")
    messages: list[HistoryMessage]


@router.post("/admin/chat", response_model=ChatResponse)
def preview_chat(
    body: ChatRequest,
    x_admin_session: str = Header(default=""),
) -> ChatResponse:
    require_site_access(x_admin_session, body.site_host)
    turn = take_chat_turn(
        body.client_id,
        body.site_host,
        body.message,
        body.session_id,
        body.answer_generally,
        employee_email=body.employee_email,
        answer_style=body.answer_style,
    )
    return chat_response(turn)


@router.get("/admin/memory", response_model=list[MemoryFactResponse])
def preview_memory(
    client_id: str = Query(default="", alias="clientId"),
    site_host: str = Query(default="", alias="siteHost"),
    x_admin_session: str = Header(default=""),
) -> list[MemoryFactResponse]:
    require_site_access(x_admin_session, site_host)
    organization = find_organization_by_host(site_host)
    return [
        MemoryFactResponse(id=fact.id, fact=fact.fact, kind=fact.kind, created_at=fact.created_at)
        for fact in recall_facts(
            organization.id, client_id.strip(), load_settings().employee_fact_limit, conversation_only=True
        )
    ]


@router.delete("/admin/memory")
def preview_forget_all(
    client_id: str = Query(default="", alias="clientId"),
    site_host: str = Query(default="", alias="siteHost"),
    x_admin_session: str = Header(default=""),
) -> dict[str, int]:
    require_site_access(x_admin_session, site_host)
    organization = find_organization_by_host(site_host)
    return {"forgotten": forget_conversation_facts(organization.id, client_id)}


@router.delete("/admin/memory/{fact_id}")
def preview_forget_fact(
    fact_id: str,
    client_id: str = Query(default="", alias="clientId"),
    site_host: str = Query(default="", alias="siteHost"),
    x_admin_session: str = Header(default=""),
) -> dict[str, bool]:
    require_site_access(x_admin_session, site_host)
    organization = find_organization_by_host(site_host)
    if not forget_fact(organization.id, client_id, fact_id):
        raise HTTPException(status_code=404, detail="That remembered fact was not found.")
    return {"forgotten": True}


def stream_chat(body: ChatRequest, request: Request) -> StreamingResponse:
    if body.client_id.strip() == "":
        raise HTTPException(status_code=400, detail="Enter a question.")
    if not body.answer_generally and body.message.strip() == "":
        raise HTTPException(status_code=400, detail="Enter a question.")
    opening = "Answering from general knowledge" if body.answer_generally else "Understanding your question"
    events = TurnEvents()

    def worker() -> None:
        register_turn(body.client_id.strip(), events.cancel)
        try:
            turn = take_chat_turn(
                body.client_id,
                body.site_host,
                body.message,
                body.session_id,
                body.answer_generally,
                events,
                body.employee_email,
                body.answer_style,
            )
            events.answer(
                turn.session_id,
                turn.message_id,
                turn.answer,
                turn.citations,
                turn.answer_kind,
                turn.choices,
                turn.remembered,
                turn.created_at,
                turn.from_page,
                turn.suggestions,
            )
        except HTTPException as error:
            detail = error.detail if isinstance(error.detail, str) else "The question could not be answered."
            events.error(detail)
        except Exception:
            request_log.exception("Chat stream failed")
            events.error("The question could not be answered.")
        finally:
            release_turn(body.client_id.strip(), events.cancel)
            events.close()

    async def generate():
        thread = threading.Thread(target=worker, daemon=True)
        yield encode_event("step", {"text": opening})
        thread.start()
        while True:
            try:
                item = events.queue.get_nowait()
            except queue.Empty:
                yield ": keep-alive\n\n"
                await asyncio.sleep(0.2)
                continue
            if item is None:
                return
            yield item

    return StreamingResponse(generate(), media_type="text/event-stream", headers=SSE_HEADERS)


@router.post("/admin/chat/decline")
def preview_chat_decline(
    body: ChatRequest,
    x_admin_session: str = Header(default=""),
) -> dict[str, bool]:
    require_site_access(x_admin_session, body.site_host)
    return decline_chat(body)


@router.post("/admin/chat/feedback", response_model=FeedbackResponse)
def preview_chat_feedback(
    body: FeedbackRequest,
    x_admin_session: str = Header(default=""),
) -> FeedbackResponse:
    require_site_access(x_admin_session, body.site_host)
    return rate_answer(body)


@router.post("/admin/chat/reset", response_model=ResetResponse)
def preview_chat_reset(
    body: ChatRequest,
    x_admin_session: str = Header(default=""),
) -> ResetResponse:
    require_site_access(x_admin_session, body.site_host)
    return reset_chat(body)


@router.post("/admin/chat/stop")
def preview_chat_stop(
    body: ChatRequest,
    x_admin_session: str = Header(default=""),
) -> dict[str, bool]:
    require_site_access(x_admin_session, body.site_host)
    stop_turn(body.client_id)
    return {"stopped": True}


@router.post("/admin/chat/stream")
async def preview_chat_stream(
    body: ChatRequest,
    request: Request,
    x_admin_session: str = Header(default=""),
) -> StreamingResponse:
    require_site_access(x_admin_session, body.site_host)
    return stream_chat(body, request)


def history_response(
    session_id: str,
    client_id: str,
    site_host: str,
    employee_email: str = "",
) -> HistoryResponse:
    scope = scoped_client_id(client_id, site_host, employee_email)
    if not session_matches(session_id, scope, normalize_site_host(site_host)):
        raise HTTPException(status_code=404, detail="That conversation was not found.")
    return HistoryResponse(
        session_id=session_id,
        messages=[
            HistoryMessage(
                id=message.id,
                role=message.role,
                content=message.content,
                citations=[citation_response(citation) for citation in message.citations],
                answer_kind=message.answer_kind,
                choices=list(message.choices),
                suggestions=list(message.suggestions),
                remembered=message.remembered,
                created_at=message.created_at,
                from_page=message.from_page,
                rating=message.rating,
                feedback_note=message.feedback_note,
            )
            for message in list_messages(session_id)
        ],
    )


@router.get("/admin/chat/{session_id}", response_model=HistoryResponse)
def preview_history(
    session_id: str,
    client_id: str = Query(default="", alias="clientId"),
    site_host: str = Query(default="", alias="siteHost"),
    x_admin_session: str = Header(default=""),
) -> HistoryResponse:
    require_site_access(x_admin_session, site_host)
    return history_response(session_id, client_id, site_host)
