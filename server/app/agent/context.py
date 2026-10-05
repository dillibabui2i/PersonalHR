import threading
from contextvars import ContextVar
from dataclasses import dataclass, field

from app.activity.trace import TurnTrace
from app.memory.service import EmployeeFact
from app.rag.types import RagAnswer, RetrievedChunk


@dataclass
class TurnContext:
    organization_id: str
    client_id: str
    session_id: str
    document_ids: frozenset[str]
    trace: TurnTrace | None
    page_text: str | None
    date_note: str
    conversation_text: str = ""
    chunks: list[RetrievedChunk] = field(default_factory=list)
    searched: bool = False
    search_failed: bool = False
    search_text: str = ""
    composed: RagAnswer | None = None
    accept_results: bool = True
    policy_unavailable: bool = False
    stated_facts: list[str] = field(default_factory=list)
    employee_facts: list[EmployeeFact] = field(default_factory=list)
    employee_facts_text: str = ""
    greeting: str = ""
    answer_style: str = "brief"
    portal_enabled: bool = True
    memory_enabled: bool = True
    portal_blocked: bool = False
    memory_blocked: bool = False
    portal_direct: str = ""
    result_lock: threading.Lock = field(default_factory=threading.Lock)


active_turn: ContextVar[TurnContext | None] = ContextVar("active_turn", default=None)
