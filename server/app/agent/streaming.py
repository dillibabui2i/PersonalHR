import json
import queue
import threading
from typing import Any

from app.rag.types import Citation

STEP_LABELS = {
    "understand_turn": "Understanding your question",
    "page_check": "Understanding your question",
    "policy_researcher": "Searching your policies",
    "search_documents": "Searching your policies",
    "search_conversation": "Looking through this conversation",
    "lookup_term": "Searching your policies",
    "calculate_leave_days": "Counting the days",
    "count_working_days": "Counting the days",
    "answer_composer": "Writing the answer",
    "answer": "Writing the answer",
    "Guardrail passed": "Checking the answer",
    "Passed after retry": "Checking the answer",
    "Blocked": "Checking the answer",
    "general answer": "Answering from general knowledge",
}

SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def step_label(name: str) -> str:
    return STEP_LABELS.get(name, "")


def encode_event(name: str, payload: dict[str, Any]) -> str:
    return f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


class CancelToken:
    def __init__(self) -> None:
        self._event = threading.Event()
        self._response: Any = None
        self._lock = threading.Lock()

    def is_set(self) -> bool:
        return self._event.is_set()

    def bind(self, response: object) -> None:
        with self._lock:
            self._response = response
            if self._event.is_set():
                self._close()

    def stop(self) -> None:
        self._event.set()
        with self._lock:
            self._close()

    def _close(self) -> None:
        response = self._response
        if response is None:
            return
        try:
            response.close()
        except Exception:
            return


_active_turns: dict[str, CancelToken] = {}
_active_lock = threading.Lock()


def register_turn(client_id: str, token: CancelToken) -> None:
    with _active_lock:
        _active_turns[client_id] = token


def release_turn(client_id: str, token: CancelToken) -> None:
    with _active_lock:
        if _active_turns.get(client_id) is token:
            _active_turns.pop(client_id, None)


def stop_turn(client_id: str) -> None:
    with _active_lock:
        token = _active_turns.get(client_id.strip())
    if token is not None:
        token.stop()


class TurnEvents:
    def __init__(self) -> None:
        self.queue: queue.Queue[str | None] = queue.Queue()
        self.cancel = CancelToken()
        self.session_id = ""

    def remember_session(self, session_id: str, text: str) -> None:
        self.session_id = session_id
        self.step(text)

    def step(self, text: str) -> None:
        payload: dict[str, Any] = {"text": text}
        if self.session_id != "":
            payload["sessionId"] = self.session_id
        self.queue.put(encode_event("step", payload))

    def token(self, text: str) -> None:
        if text == "":
            return
        self.queue.put(encode_event("token", {"text": text}))

    def answer(
        self,
        session_id: str,
        message_id: str,
        answer: str,
        citations: list[Citation],
        answer_kind: str,
        choices: tuple[str, ...] = (),
        remembered: bool = False,
        created_at: str = "",
        from_page: bool = False,
        suggestions: tuple[str, ...] = (),
    ) -> None:
        self.queue.put(
            encode_event(
                "answer",
                {
                    "sessionId": session_id,
                    "messageId": message_id,
                    "answer": answer,
                    "citations": [
                        {
                            "documentName": citation.document_name,
                            "sectionTitle": citation.section_title,
                            "excerpt": citation.excerpt,
                        }
                        for citation in citations
                    ],
                    "answerKind": answer_kind,
                    "choices": list(choices),
                    "remembered": remembered,
                    "createdAt": created_at,
                    "fromPage": from_page,
                    "suggestions": list(suggestions),
                },
            )
        )

    def error(self, detail: str) -> None:
        self.queue.put(encode_event("error", {"detail": detail}))

    def close(self) -> None:
        self.queue.put(None)
