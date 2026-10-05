import logging
import os
import re
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import replace
from contextvars import copy_context
from datetime import date, datetime

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from crewai.flow import ConversationConfig, ConversationState, Flow, listen
from crewai.flow.persistence import persist

from app.activity.trace import open_step, raise_if_cancelled, shorten_summary
from app.agent.agents import require_hr_agents
from app.agent.context import TurnContext, active_turn
from app.agent.persistence import flow_store
from app.agent.guardrail import (
    PASSED_REASON,
    entitlement_question,
    grounding_failure,
    leave_count_answer,
    leave_gap_answer,
)
from app.agent.planner import CONVERSATION_QUERY, MEMORY_QUERY, TurnPlan, plan_payload, understand_turn
from app.organizations.assistant import read_assistant_settings, limit_plan
from app.chat.sessions import recent_turns
from app.agent.tools.dates import CalculateLeaveDaysTool, CountWorkingDaysTool
from app.agent.tools.lookup_term import LookupTermTool
from app.agent.suggestions import follow_up_questions
from app.agent.tools.conversation import SearchConversationTool
from app.agent.tools.search_documents import SearchDocumentsTool
from app.cache.service import read_cached_answer, store_cached_answer
from app.core.settings import load_settings
from app.memory.service import (
    facts_context,
    mark_facts_used,
    recall_facts,
    refuses_memory,
)
from app.rag.generation import generate_answer, generate_greeting
from app.rag.prompts import CONVERSATION_MISS, MEMORY_DISABLED_REPLY, NO_ANSWER, UNAVAILABLE_ANSWER
from app.rag.types import RagAnswer

agent_log = logging.getLogger("personal_hr.agents")
STATED_BALANCE = re.compile(r"\bi have \d+", re.IGNORECASE)
BRANCH_WAIT_SECONDS = 60
MEMORY_REFUSAL = "I can't remember passwords, passcodes, tokens, or other secrets."


class HrChatState(ConversationState):
    organization_id: str = ""
    standalone_question: str = ""
    route: str = "ANSWER"
    needs_policy: bool = True
    needs_portal: bool = False
    needs_memory: bool = False
    needs_conversation: bool = False
    needs_dates: bool = False
    start_date: str = ""
    end_date: str = ""
    clarifying_question: str = ""
    choice_labels: str = ""


@ConversationConfig(defer_trace_finalization=True)
class HrChatFlow(Flow[HrChatState]):
    def route_turn(self, context: object) -> str:
        del context
        turn = active_turn.get()
        question = (self.state.current_user_message or "").strip()
        if turn is None or question == "":
            return "OUT_OF_SCOPE"
        raise_if_cancelled(turn.trace)
        self.load_assistant_settings(turn)
        skip_cache = (
            MEMORY_QUERY.search(question) is not None
            or CONVERSATION_QUERY.search(question) is not None
        )
        if turn.trace is not None and not skip_cache:
            cached = read_cached_answer(turn.organization_id, turn.document_ids, question, turn.answer_style)
            if cached is not None:
                self.state.standalone_question = question
                self.state.route = "ANSWER"
                self.state.needs_policy = True
                turn.composed = self.with_suggestions(turn, question, cached)
                turn.trace.cached = True
                turn.trace.set_plan(
                    "ANSWER",
                    question,
                    {"standaloneQuestion": question, "route": "ANSWER", "cacheHit": True},
                )
                return "CACHED"
        with open_step(turn.trace, "understand_turn", question, "plan") as step:
            plan = understand_turn(
                question,
                recent_turns(turn.session_id, load_settings().history_turns),
                turn.organization_id,
                turn.client_id,
                turn.page_text or "",
                turn.memory_enabled,
            )
            limits = limit_plan(plan, read_assistant_settings(turn.organization_id))
            turn.portal_blocked = limits.portal_blocked
            turn.memory_blocked = limits.memory_blocked
            self.apply_plan(turn, plan)
            dates = "none" if plan.start_date == "" else f"{plan.start_date} to {plan.end_date}"
            step.output_summary = (
                f"Route {plan.route}; policy {plan.needs_policy}; conversation {plan.needs_conversation}; "
                f"portal {plan.needs_portal}; dates {dates}; standalone: {plan.standalone_question}"
            )
        agent_log.info("planner returned route %s to flow", plan.route)
        return plan.route

    def apply_plan(self, context: TurnContext, plan: TurnPlan) -> None:
        self.state.standalone_question = plan.standalone_question
        self.state.route = plan.route
        self.state.needs_policy = plan.needs_policy
        self.state.needs_portal = plan.needs_portal
        self.state.needs_memory = plan.needs_memory
        self.state.needs_conversation = plan.needs_conversation
        self.state.needs_dates = plan.needs_dates
        self.state.start_date = plan.start_date
        self.state.end_date = plan.end_date
        self.state.clarifying_question = plan.clarifying_question
        self.state.choice_labels = "|".join(plan.choices)
        context.stated_facts = plan.stated_facts
        if context.trace is not None:
            context.trace.set_plan(plan.route, plan.standalone_question, plan_payload(plan))

    def load_assistant_settings(self, context: TurnContext) -> None:
        settings = read_assistant_settings(context.organization_id)
        context.portal_enabled = settings.portal_enabled
        context.memory_enabled = settings.memory_enabled
        if not settings.portal_enabled:
            context.page_text = None
        self.load_core_memory(context)

    def load_core_memory(self, context: TurnContext) -> None:
        """Pin durable employee facts into every turn, Letta-style core memory."""
        if not context.memory_enabled:
            context.employee_facts = []
            context.employee_facts_text = ""
            return
        context.employee_facts = recall_facts(
            context.organization_id,
            context.client_id,
            load_settings().employee_fact_limit,
        )
        context.employee_facts_text = facts_context(context.employee_facts)

    @listen("GREETING")
    @persist(flow_store())
    def handle_greeting(self) -> str:
        """Reply to greetings with the chat model, without searching documents."""
        context = active_turn.get()
        question = (self.state.current_user_message or "Hello").strip()
        facts = "" if context is None else context.employee_facts_text
        trace = None if context is None else context.trace
        result = generate_greeting(question, facts, trace)
        if context is not None:
            with open_step(context.trace, "Guardrail passed", "Conversational reply.", "guardrail") as step:
                step.output_summary = "Conversational reply."
            context.composed = replace(result, guardrail="Guardrail passed")
        self.append_assistant_message(result.answer)
        return result.answer

    @listen("OUT_OF_SCOPE")
    @persist(flow_store())
    def handle_out_of_scope(self) -> str:
        """Decline requests unrelated to the organization or the employee."""
        context = active_turn.get()
        if context is not None:
            checked = self.guard_answer(context, None, RagAnswer(answer=NO_ANSWER, citations=[]), "", False)
            context.composed = checked
            self.append_assistant_message(checked.answer)
            return checked.answer
        self.append_assistant_message(NO_ANSWER)
        return NO_ANSWER

    @listen("RESET")
    @persist(flow_store())
    def handle_reset(self) -> str:
        """End this conversation so the next question starts without its history."""
        context = active_turn.get()
        if context is not None:
            with open_step(context.trace, "Reset", "New conversation.", "plan") as step:
                step.output_summary = "Session ended."
            context.composed = RagAnswer(answer="", citations=[])
        return ""

    @listen("CLARIFY")
    @persist(flow_store())
    def handle_clarify(self) -> str:
        """Ask one short question when the request could mean more than one documented answer."""
        context = active_turn.get()
        question = self.state.clarifying_question.strip() or "Which kind of leave do you mean?"
        choices = tuple(label for label in self.state.choice_labels.split("|") if label != "")
        if context is not None:
            with open_step(context.trace, "Guardrail passed", "Clarifying question.", "guardrail") as step:
                step.output_summary = "Clarifying question."
            context.composed = RagAnswer(answer=question, citations=[], guardrail="Guardrail passed", choices=choices)
        self.append_assistant_message(question)
        return question

    @listen("ANSWER")
    @persist(flow_store())
    def handle_answer(self) -> str:
        """Run only the specialists selected by the turn plan, then compose from their findings."""
        context = active_turn.get()
        if context is not None and context.memory_blocked:
            return self.publish_fixed(context, MEMORY_DISABLED_REPLY)
        require_hr_agents()
        if context is not None:
            context.date_note = ""
        raise_if_cancelled(None if context is None else context.trace)
        question = self.state.standalone_question.strip()
        if context is None or question == "":
            self.append_assistant_message(NO_ANSWER)
            return NO_ANSWER
        current_message = (self.state.current_user_message or "").strip()
        remembered = context.stated_facts if context.memory_enabled else []
        memory_only = (
            remembered != []
            and not self.state.needs_policy
            and not self.state.needs_portal
            and not self.state.needs_dates
            and not self.state.needs_memory
        )
        if context.memory_enabled and refuses_memory(current_message, remembered):
            result = RagAnswer(answer=MEMORY_REFUSAL, citations=[], guardrail="Guardrail passed")
            context.composed = result
            self.append_assistant_message(result.answer)
            return result.answer
        if memory_only:
            shared = remembered[0]
            result = RagAnswer(
                answer=f"I'll remember that {self.memory_acknowledgement(shared)}.",
                citations=[],
                guardrail="Guardrail passed",
            )
            context.composed = result
            self.append_assistant_message(result.answer)
            return result.answer
        cached = self.cached_policy_answer(context, question)
        if cached is not None:
            context.composed = cached
            self.append_assistant_message(cached.answer)
            return cached.answer
        result = self.answer_with_specialists(context, question)
        self.cache_policy_answer(context, question, result)
        context.composed = result
        self.append_assistant_message(result.answer)
        return result.answer

    @listen("CACHED")
    @persist(flow_store())
    def handle_cached(self) -> str:
        """Return an unchanged policy answer without running the planner or specialists."""
        context = active_turn.get()
        if context is None or context.composed is None:
            self.append_assistant_message(NO_ANSWER)
            return NO_ANSWER
        with open_step(context.trace, "Answer cache", self.state.standalone_question, "cache") as step:
            step.output_summary = "Cached answer returned; documents are unchanged."
            if context.trace is not None and context.trace.on_token is not None:
                context.trace.on_token(context.composed.answer)
        self.append_assistant_message(context.composed.answer)
        return context.composed.answer

    def search_conversation(self, context: TurnContext) -> None:
        query = (self.state.current_user_message or self.state.standalone_question).strip()
        context.conversation_text = SearchConversationTool().execute(query)

    def conversation_only(self, context: TurnContext) -> bool:
        return (
            self.state.needs_conversation
            and not self.state.needs_policy
            and not self.state.needs_dates
            and context.stated_facts == []
        )

    def policy_cache_eligible(self) -> bool:
        context = active_turn.get()
        return (
            self.state.needs_policy
            and not self.state.needs_portal
            and not self.state.needs_memory
            and not self.state.needs_conversation
            and not self.state.needs_dates
            and (context is None or context.stated_facts == [])
        )

    def cached_policy_answer(self, context: TurnContext, question: str) -> RagAnswer | None:
        if context.trace is None or not self.policy_cache_eligible():
            return None
        cached = read_cached_answer(context.organization_id, context.document_ids, question, context.answer_style)
        if cached is None:
            return None
        cached = self.with_suggestions(context, question, cached)
        with open_step(context.trace, "Answer cache", question, "cache") as step:
            step.output_summary = "Cached answer returned; documents are unchanged."
            if context.trace is not None:
                context.trace.cached = True
                if context.trace.on_token is not None:
                    context.trace.on_token(cached.answer)
        return cached

    def cache_policy_answer(self, context: TurnContext, question: str, result: RagAnswer) -> None:
        if (
            context.trace is None
            or not self.policy_cache_eligible()
            or result.answer in (NO_ANSWER, UNAVAILABLE_ANSWER)
            or result.guardrail == "Blocked"
        ):
            return
        store_cached_answer(context.organization_id, context.document_ids, question, result, context.answer_style)

    def answer_with_specialists(self, context: TurnContext, question: str) -> RagAnswer:
        needs_policy = self.state.needs_policy
        needs_dates = self.state.needs_dates and self.state.start_date != "" and self.state.end_date != ""
        if self.state.needs_conversation:
            self.search_conversation(context)
        self.load_core_memory(context)
        branches = [name for name, needed in (("policy", needs_policy), ("dates", needs_dates)) if needed]
        agent_log.info("flow is calling %s", " and ".join(branches) if branches else "no specialist")
        if len(branches) >= 2:
            page_facts = self.run_together(context, question, branches)
        elif needs_policy:
            self.research_policy(context, question)
            page_facts = None
        elif needs_dates:
            self.run_date_tools(context)
            page_facts = None
        else:
            page_facts = None
        self.note_unavailable(context, page_facts)
        today = datetime.now().astimezone().strftime("Today is %A, %d %B %Y.")
        context.date_note = today if context.date_note == "" else f"{today}\n{context.date_note}"
        if self.conversation_only(context) and context.conversation_text == "":
            agent_log.info("flow returned no conversation match")
            return self.guard_answer(
                context,
                page_facts,
                RagAnswer(answer=CONVERSATION_MISS, citations=[]),
                question,
                False,
            )
        if context.search_failed and page_facts is None and context.date_note == "":
            agent_log.info("flow returned unavailable")
            return self.guard_answer(context, page_facts, RagAnswer(answer=UNAVAILABLE_ANSWER, citations=[]), question, False)
        result = self.compose_answer(context, question, page_facts)
        if self.state.needs_memory and context.employee_facts != [] and result.answer != NO_ANSWER:
            mark_facts_used(context.employee_facts)
        return result

    def run_together(self, context: TurnContext, question: str, branches: list[str]) -> str | None:
        """Start policy and date work together. A branch that runs past the limit is unavailable."""
        pool = ThreadPoolExecutor(max_workers=len(branches))
        futures: dict[str, Future[object]] = {}
        if "policy" in branches:
            futures["policy"] = pool.submit(copy_context().run, self.research_policy, context, question)
        if "dates" in branches:
            futures["dates"] = pool.submit(copy_context().run, self.run_date_tools, context)
        done, _pending = wait(set(futures.values()), timeout=BRANCH_WAIT_SECONDS)
        page_facts = self.collect_branches(context, futures, done)
        pool.shutdown(wait=False, cancel_futures=True)
        return page_facts

    def collect_branches(
        self,
        context: TurnContext,
        futures: dict[str, Future[object]],
        done: set[Future[object]],
    ) -> str | None:
        policy = futures.get("policy")
        if policy is not None:
            self.finish_policy(context, policy, policy in done)
        page_facts = None
        dates = futures.get("dates")
        if dates is not None and dates not in done and not dates.done() and context.date_note == "":
            agent_log.info("date tools returned unavailable to flow")
            context.date_note = "Date counting is unavailable."
        elif dates is not None and (dates in done or dates.done()):
            dates.result()
        return page_facts if isinstance(page_facts, str) or page_facts is None else None

    def finish_policy(self, context: TurnContext, policy: Future[object], finished: bool) -> None:
        if finished or policy.done():
            policy.result()
            return
        with context.result_lock:
            arrived = policy.done() or context.chunks != []
            if not arrived:
                context.accept_results = False
                context.policy_unavailable = True
                context.chunks = []
        if arrived and policy.done():
            policy.result()
            return
        if not arrived:
            agent_log.info("policy_researcher returned unavailable to flow")

    def note_unavailable(self, context: TurnContext, page_facts: str | None) -> None:
        notes: list[str] = []
        if context.policy_unavailable:
            notes.append("Policy search is unavailable.")
        if notes == []:
            return
        extra = " ".join(notes)
        context.date_note = extra if context.date_note == "" else f"{context.date_note}\n{extra}"

    def run_date_tools(self, context: TurnContext) -> None:
        if not self.state.needs_dates or self.state.start_date == "" or self.state.end_date == "":
            return
        raise_if_cancelled(context.trace)
        agent_log.info("flow is calling calculate_leave_days")
        leave_days = CalculateLeaveDaysTool().execute(self.state.start_date, self.state.end_date)
        agent_log.info("calculate_leave_days returned a count to flow")
        agent_log.info("flow is calling count_working_days")
        working_days = CountWorkingDaysTool().execute(date.today().isoformat(), self.state.start_date)
        agent_log.info("count_working_days returned a count to flow")
        context.date_note = (
            f"{leave_days}\n{working_days}\n"
            "workingDaysBetween counts weekdays after fromDate and before toDate. "
            "The notice count uses today as fromDate and the leave start as toDate."
        )

    def research_policy(self, context: TurnContext, question: str) -> None:
        raise_if_cancelled(context.trace)
        with open_step(context.trace, "policy_researcher", question, "agent") as researcher:
            search_question = LookupTermTool().expand_question(question)
            agent_log.info("policy_researcher is calling search_documents")
            SearchDocumentsTool().execute(search_question)
            researcher.output_summary = f"{len(context.chunks)} excerpts"
        self.log_search_return(context)

    def log_search_return(self, context: TurnContext) -> None:
        if context.search_failed:
            agent_log.info("search_documents returned a failure to policy_researcher")
            agent_log.info("policy_researcher returned a failure to flow")
            return
        if context.chunks == []:
            agent_log.info("search_documents returned no excerpts to policy_researcher")
            agent_log.info("policy_researcher returned no excerpts to flow")
            return
        count = len(context.chunks)
        agent_log.info("search_documents returned %d excerpts to policy_researcher", count)
        agent_log.info("policy_researcher returned %d excerpts to flow", count)

    def findings_note(self, context: TurnContext, question: str) -> str:
        note = self.stated_balance_note(question, context.date_note)
        if context.conversation_text == "":
            return note
        return context.conversation_text if note == "" else f"{note}\n{context.conversation_text}"

    def stated_balance_note(self, question: str, date_note: str) -> str:
        if STATED_BALANCE.search(question) is None:
            return date_note
        note = f"The employee stated: {question}"
        return note if date_note == "" else f"{date_note}\n{note}"

    def compose_answer(self, context: TurnContext, question: str, page_facts: str | None) -> RagAnswer:
        if (
            page_facts is None
            and context.chunks == []
            and context.date_note == ""
            and self.answer_facts(context) == ""
            and context.conversation_text == ""
        ):
            agent_log.info("flow returned no answer")
            return self.guard_answer(context, page_facts, RagAnswer(answer=NO_ANSWER, citations=[]), question, False)
        raise_if_cancelled(context.trace)
        sources = self.composer_inputs(context, page_facts)
        agent_log.info("flow is calling answer_composer with %s", sources)
        with open_step(context.trace, "answer_composer", f"{question} ({sources})", "agent") as composer:
            user_message = (self.state.current_user_message or question).strip()
            counted = leave_count_answer(question, context.chunks) or leave_count_answer(
                user_message, context.chunks
            )
            ask_count = entitlement_question(question) or entitlement_question(user_message)
            if counted is not None and ask_count:
                agent_log.info("answer_composer used extracted leave entitlement")
                result = counted
            else:
                agent_log.info(
                    "answer_composer calling the model; extracted=%s entitlement=%s",
                    counted is not None,
                    ask_count,
                )
                result = generate_answer(
                    question,
                    context.chunks,
                    page_facts,
                    context.date_note,
                    context.trace,
                    record_step=False,
                    employee_facts=self.answer_facts(context),
                    answer_style=context.answer_style,
                    conversation=context.conversation_text,
                )
            result, status, reason = self.review_answer(context, question, page_facts, result, True)
            composer.output_summary = result.answer
        if result.answer == NO_ANSWER:
            agent_log.info("answer_composer returned no answer to flow")
        else:
            agent_log.info("answer_composer returned an answer to flow")
        checked = self.publish_guardrail(context, result, status, reason)
        return self.with_suggestions(context, question, checked)

    def guard_answer(
        self,
        context: TurnContext,
        page_facts: str | None,
        result: RagAnswer,
        question: str,
        allow_retry: bool,
    ) -> RagAnswer:
        reviewed, status, reason = self.review_answer(context, question, page_facts, result, allow_retry)
        return self.publish_guardrail(context, reviewed, status, reason)

    def review_answer(
        self,
        context: TurnContext,
        question: str,
        page_facts: str | None,
        result: RagAnswer,
        allow_retry: bool,
    ) -> tuple[RagAnswer, str, str]:
        reply = result.answer.strip()
        if reply == MEMORY_DISABLED_REPLY:
            return result, "Guardrail passed", PASSED_REASON
        if result.answer.strip() in {"", NO_ANSWER}:
            counted = leave_count_answer(question, context.chunks)
            if counted is not None:
                result = counted
            else:
                gap = leave_gap_answer(question, context.chunks)
                if gap is not None:
                    result = gap
        else:
            gap = leave_gap_answer(question, context.chunks)
            if gap is not None:
                result = gap
        tool_text = self.findings_note(context, question)
        failure = grounding_failure(
            result.answer,
            result.citations,
            context.chunks,
            None,
            page_facts,
            tool_text,
            self.answer_facts(context),
        )
        if failure is None:
            return result, "Guardrail passed", PASSED_REASON
        counted = leave_count_answer(question, context.chunks)
        if counted is not None:
            extracted_failure = grounding_failure(
                counted.answer,
                counted.citations,
                context.chunks,
                None,
                page_facts,
                tool_text,
                self.answer_facts(context),
            )
            if extracted_failure is None:
                agent_log.info("guardrail used extracted leave entitlement after a composer miss")
                return counted, "Guardrail passed", PASSED_REASON
        if not allow_retry:
            return RagAnswer(answer=NO_ANSWER, citations=[]), "Blocked", failure
        agent_log.info("guardrail is calling answer_composer for a retry")
        retried = generate_answer(
            question,
            context.chunks,
            page_facts,
            context.date_note,
            context.trace,
            record_step=False,
            correction=failure,
            employee_facts=self.answer_facts(context),
            answer_style=context.answer_style,
            conversation=context.conversation_text,
        )
        second = grounding_failure(
            retried.answer,
            retried.citations,
            context.chunks,
            None,
            page_facts,
            self.findings_note(context, question),
            self.answer_facts(context),
        )
        if second is None:
            return retried, "Passed after retry", failure
        return RagAnswer(answer=NO_ANSWER, citations=[]), "Blocked", second

    def publish_fixed(self, context: TurnContext, answer: str) -> str:
        with open_step(context.trace, "Assistant settings", answer, "plan") as step:
            step.output_summary = answer
        result = RagAnswer(answer=answer, citations=[], guardrail="Guardrail passed")
        context.composed = result
        self.append_assistant_message(answer)
        return answer

    def publish_guardrail(self, context: TurnContext, result: RagAnswer, status: str, reason: str) -> RagAnswer:
        with open_step(context.trace, status, reason, "guardrail") as step:
            step.output_summary = reason
        agent_log.info("guardrail returned %s to flow", status)
        return replace(result, guardrail=status)

    def with_suggestions(self, context: TurnContext, question: str, result: RagAnswer) -> RagAnswer:
        if result.choices != () or result.answer in {NO_ANSWER, UNAVAILABLE_ANSWER}:
            return result
        document_ids = frozenset(chunk.document_id for chunk in context.chunks if chunk.document_id != "")
        if document_ids == frozenset():
            if result.citations == []:
                return result
            document_ids = context.document_ids
        suggestions = follow_up_questions(context.organization_id, document_ids, question, result.answer)
        if suggestions == ():
            return result
        return replace(result, suggestions=suggestions)

    def answer_facts(self, context: TurnContext) -> str:
        """Personal facts belong in an answer only when the question is about the employee."""
        if not self.state.needs_memory:
            return ""
        return context.employee_facts_text

    def composer_inputs(self, context: TurnContext, page_facts: str | None) -> str:
        parts: list[str] = []
        if context.chunks != []:
            parts.append("documents")
        if page_facts:
            parts.append("page facts")
        if context.date_note != "":
            parts.append("tool results")
        if self.answer_facts(context) != "":
            parts.append("employee-stated facts")
        if context.conversation_text != "":
            parts.append("conversation")
        if parts == []:
            return "nothing"
        return " and ".join(parts)

    def memory_acknowledgement(self, fact: str) -> str:
        cleaned = fact.strip().rstrip(".")
        if cleaned.lower().startswith("i "):
            return f"you {cleaned[2:]}"
        if cleaned.lower().startswith("my "):
            return f"your {cleaned[3:]}"
        return cleaned
