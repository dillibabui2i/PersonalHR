import json
import threading
import time
from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Iterator, Protocol

from app.chat.cancel import TurnCancelled


class Cancellation(Protocol):
    def is_set(self) -> bool: ...

    def bind(self, response: object) -> None: ...

SUMMARY_LIMIT = 200


def shorten_summary(text: str) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= SUMMARY_LIMIT:
        return cleaned
    return f"{cleaned[: SUMMARY_LIMIT - 1].rstrip()}…"


@dataclass
class StepOutcome:
    status: str = "done"
    output_summary: str = ""


@dataclass
class StepDraft:
    kind: str
    name: str
    input_summary: str
    output_summary: str
    status: str
    started_ms: int
    ended_ms: int


@dataclass
class TurnTrace:
    session_id: str
    organization_id: str
    client_id: str
    question: str
    standalone_question: str = ""
    route: str = "pipeline"
    plan_json: str = "{}"
    engine: str = "pipeline"
    run_id: str = ""
    started: float = field(default_factory=time.perf_counter)
    steps: list[StepDraft] = field(default_factory=list)
    step_lock: threading.Lock = field(default_factory=threading.Lock)
    on_step: Callable[[StepDraft], None] | None = None
    on_step_start: Callable[[str], None] | None = None
    on_plan: Callable[[], None] | None = None
    on_token: Callable[[str], None] | None = None
    cancel: Cancellation | None = None
    streamed_tokens: bool = False
    first_token_ms: int | None = None
    cached: bool = False

    def __post_init__(self) -> None:
        if self.standalone_question == "":
            self.standalone_question = self.question

    def elapsed_ms(self) -> int:
        return int((time.perf_counter() - self.started) * 1000)

    def set_plan(self, route: str, standalone_question: str, plan: dict[str, object]) -> None:
        self.route = route
        self.standalone_question = standalone_question
        self.plan_json = json.dumps(plan)
        if self.on_plan is not None:
            self.on_plan()

    @contextmanager
    def step(self, name: str, input_summary: str, kind: str = "pipeline") -> Iterator[StepOutcome]:
        if self.on_step_start is not None:
            self.on_step_start(name)
        started_ms = self.elapsed_ms()
        outcome = StepOutcome()
        try:
            yield outcome
        except TurnCancelled:
            outcome.status = "stopped"
            if outcome.output_summary == "":
                outcome.output_summary = "Stopped"
            raise
        except Exception:
            outcome.status = "failed"
            if outcome.output_summary == "":
                outcome.output_summary = "This step failed."
            raise
        finally:
            draft = StepDraft(
                kind=kind,
                name=name,
                input_summary=shorten_summary(input_summary),
                output_summary=shorten_summary(outcome.output_summary),
                status=outcome.status,
                started_ms=started_ms,
                ended_ms=self.elapsed_ms(),
            )
            with self.step_lock:
                self.steps.append(draft)
                if self.on_step is not None:
                    self.on_step(draft)


def raise_if_cancelled(trace: TurnTrace | None) -> None:
    if trace is None or trace.cancel is None:
        return
    if trace.cancel.is_set():
        raise TurnCancelled()


def open_step(
    trace: TurnTrace | None,
    name: str,
    input_summary: str,
    kind: str = "pipeline",
) -> AbstractContextManager[StepOutcome]:
    if trace is None:
        return nullcontext(StepOutcome())
    return trace.step(name, input_summary, kind)
