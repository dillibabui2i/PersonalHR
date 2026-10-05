from datetime import date
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from app.activity.trace import open_step
from app.agent.context import active_turn
from app.rag.date_math import calendar_days_inclusive, count_working_days


class LeaveDaysInput(BaseModel):
    start_date: str = Field(alias="startDate", description="First date, YYYY-MM-DD.")
    end_date: str = Field(alias="endDate", description="Last date, YYYY-MM-DD.")

    model_config = {"populate_by_name": True}


class WorkingDaysInput(BaseModel):
    from_date: str = Field(alias="fromDate", description="Earlier date, YYYY-MM-DD.")
    to_date: str = Field(alias="toDate", description="Later date, YYYY-MM-DD.")

    model_config = {"populate_by_name": True}


def dated(value: date) -> str:
    return f"{value.isoformat()} ({value.strftime('%A %-d %B %Y')})"


class CalculateLeaveDaysTool(BaseTool):
    name: str = "calculate_leave_days"
    description: str = "Count calendar days from startDate through endDate. This does not approve leave."
    args_schema: Type[BaseModel] = LeaveDaysInput

    def _run(self, start_date: str, end_date: str) -> str:
        return self.execute(start_date, end_date)

    def execute(self, start_date: str, end_date: str) -> str:
        context = active_turn.get()
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)
        summary = f"{start.isoformat()} to {end.isoformat()}"
        with open_step(None if context is None else context.trace, "calculate_leave_days", summary, "tool") as outcome:
            if end < start:
                outcome.output_summary = "The end date is before the start date."
                return outcome.output_summary
            text = "\n".join(
                [
                    f"startDate: {dated(start)}",
                    f"endDate: {dated(end)}",
                    f"calendarDaysInclusive: {calendar_days_inclusive(start, end)}",
                ]
            )
            outcome.output_summary = text.replace("\n", "; ")
            return text


class CountWorkingDaysTool(BaseTool):
    name: str = "count_working_days"
    description: str = "Count weekdays strictly between fromDate and toDate. This does not approve leave."
    args_schema: Type[BaseModel] = WorkingDaysInput

    def _run(self, from_date: str, to_date: str) -> str:
        return self.execute(from_date, to_date)

    def execute(self, from_date: str, to_date: str) -> str:
        context = active_turn.get()
        start = date.fromisoformat(from_date)
        end = date.fromisoformat(to_date)
        summary = f"{start.isoformat()} to {end.isoformat()}"
        with open_step(None if context is None else context.trace, "count_working_days", summary, "tool") as outcome:
            if end < start:
                outcome.output_summary = "The end date is before the start date."
                return outcome.output_summary
            text = "\n".join(
                [
                    f"fromDate: {dated(start)}",
                    f"toDate: {dated(end)}",
                    f"workingDaysBetween: {count_working_days(start, end)}",
                ]
            )
            outcome.output_summary = text.replace("\n", "; ")
            return text
