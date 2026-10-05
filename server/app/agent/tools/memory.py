from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from app.agent.context import active_turn
from app.core.settings import load_settings
from app.memory.service import fact_kind, facts_context, recall_facts, save_fact


class SaveFactInput(BaseModel):
    fact: str = Field(description="One fact the employee stated about themself.")
    kind: str = Field(default="general", description="Fact category.")


class NoMemoryInput(BaseModel):
    pass


class RecallEmployeeFactsTool(BaseTool):
    name: str = "recall_employee_facts"
    description: str = "Recall facts this browser's employee previously chose to share for this organization."
    args_schema: Type[BaseModel] = NoMemoryInput

    def _run(self) -> str:
        return self.execute()

    def execute(self) -> str:
        context = active_turn.get()
        if context is None:
            return ""
        facts = recall_facts(
            context.organization_id,
            context.client_id,
            load_settings().employee_fact_limit,
        )
        return facts_context(facts)


class SaveEmployeeFactTool(BaseTool):
    name: str = "save_employee_fact"
    description: str = "Save one non-secret fact the employee explicitly shared about themself."
    args_schema: Type[BaseModel] = SaveFactInput

    def _run(self, fact: str, kind: str = "general") -> str:
        return self.execute(fact, kind)

    def execute(self, fact: str, kind: str = "general") -> str:
        context = active_turn.get()
        if context is None:
            return "refused"
        saved = save_fact(
            context.organization_id,
            context.client_id,
            fact,
            kind if kind != "" else fact_kind(fact),
            "",
            context.page_text,
        )
        return "saved" if saved else "refused"
