import logging
import re
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from app.activity.trace import open_step
from app.agent.context import active_turn
from app.rag.glossary import spell_out_terms
from app.rag.vector_store import read_terms

agent_log = logging.getLogger("personal_hr.agents")


class LookupInput(BaseModel):
    term: str = Field(description="The abbreviation to look up in this organization's documents.")


class LookupTermTool(BaseTool):
    name: str = "lookup_term"
    description: str = (
        "Look up an abbreviation in this organization's uploaded documents. "
        "Use it before searching when the question uses a short form."
    )
    args_schema: Type[BaseModel] = LookupInput

    def _run(self, term: str) -> str:
        return self.execute(term)

    def execute(self, term: str) -> str:
        context = active_turn.get()
        if context is None:
            return "unknown"
        agent_log.info("policy_researcher is calling lookup_term")
        long_form = ""
        for short_form, expanded in read_terms(context.organization_id).items():
            if short_form.lower() == term.strip().lower():
                long_form = expanded
                break
        with open_step(context.trace, "lookup_term", term.strip(), "tool") as outcome:
            if long_form == "":
                outcome.output_summary = "unknown"
                agent_log.info("lookup_term returned unknown to policy_researcher")
                return "unknown"
            outcome.output_summary = long_form
            agent_log.info("lookup_term returned a long form to policy_researcher")
            return long_form

    def expand_question(self, question: str) -> str:
        context = active_turn.get()
        if context is None:
            return question
        matched: dict[str, str] = {}
        for short_form, long_form in read_terms(context.organization_id).items():
            mentioned = re.search(rf"\b{re.escape(short_form)}\b", question, flags=re.IGNORECASE)
            if mentioned is None or long_form.lower() in question.lower():
                continue
            self.execute(short_form)
            matched[short_form] = long_form
        return spell_out_terms(question, matched)
