import uuid

from app.activity.trace import TurnTrace
from app.rag.prompts import NO_ANSWER
from app.rag.types import RagAnswer, SearchFilter


def assistant_engine_name() -> str:
    return "agent"


def answer_with_engine(
    organization_id: str,
    question: str,
    search_filter: SearchFilter,
    page_text: str | None,
    date_note: str,
    trace: TurnTrace | None,
    session_id: str,
    answer_style: str = "brief",
) -> RagAnswer:
    return answer_with_flow(
        organization_id,
        question,
        search_filter,
        page_text,
        date_note,
        trace,
        session_id,
        answer_style,
    )


def answer_with_flow(
    organization_id: str,
    question: str,
    search_filter: SearchFilter,
    page_text: str | None,
    date_note: str,
    trace: TurnTrace | None,
    session_id: str,
    answer_style: str = "brief",
) -> RagAnswer:
    from app.agent.context import TurnContext, active_turn
    from app.agent.flow import HrChatFlow
    from app.agent.persistence import flow_store

    context = TurnContext(
        organization_id=organization_id,
        client_id="" if trace is None else trace.client_id,
        session_id=session_id,
        document_ids=search_filter.document_ids or frozenset(),
        trace=trace,
        page_text=page_text,
        date_note=date_note,
        answer_style="detailed" if answer_style == "detailed" else "brief",
    )
    token = active_turn.set(context)
    flow = HrChatFlow(persistence=flow_store())
    flow.suppress_flow_events = True
    try:
        flow.handle_turn(question, session_id=session_id or str(uuid.uuid4()))
    finally:
        try:
            flow.finalize_session_traces()
        except Exception:
            pass
        active_turn.reset(token)
    if context.composed is None:
        return RagAnswer(answer=NO_ANSWER, citations=[])
    return context.composed
