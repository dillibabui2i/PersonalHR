from pathlib import Path

import yaml
from crewai import Agent, LLM

from app.agent.tools.conversation import SearchConversationTool
from app.agent.tools.dates import CalculateLeaveDaysTool, CountWorkingDaysTool
from app.agent.tools.lookup_term import LookupTermTool
from app.agent.tools.memory import SaveEmployeeFactTool
from app.agent.tools.search_documents import SearchDocumentsTool
from app.core.settings import load_settings
from app.llm.chat_model import selected_chat_model


def chat_llm() -> LLM:
    settings = load_settings()
    return LLM(
        model=f"ollama/{selected_chat_model(settings.chat_model)}",
        base_url=settings.ollama_base_url,
        temperature=0,
    )


def planner_llm() -> LLM:
    settings = load_settings()
    return LLM(
        model=f"ollama/{selected_chat_model(settings.chat_model)}",
        base_url=settings.ollama_base_url,
        temperature=0,
    )


def agent_profile(name: str) -> dict[str, str]:
    path = Path(__file__).resolve().parent / "config" / "agents.yaml"
    loaded = yaml.safe_load(path.read_text())
    profile = loaded[name]
    return {
        "role": str(profile["role"]).strip(),
        "goal": str(profile["goal"]).strip(),
        "backstory": str(profile["backstory"]).strip(),
    }


def policy_researcher() -> Agent:
    profile = agent_profile("policy_researcher")
    return Agent(
        role=profile["role"],
        goal=profile["goal"],
        backstory=profile["backstory"],
        llm=chat_llm(),
        function_calling_llm=planner_llm(),
        tools=[SearchDocumentsTool(), LookupTermTool()],
        max_iter=3,
        max_execution_time=45,
        allow_delegation=False,
        allow_code_execution=False,
        verbose=False,
    )


def answer_composer() -> Agent:
    profile = agent_profile("answer_composer")
    return Agent(
        role=profile["role"],
        goal=profile["goal"],
        backstory=profile["backstory"],
        llm=chat_llm(),
        tools=[],
        max_iter=1,
        max_execution_time=45,
        allow_delegation=False,
        allow_code_execution=False,
        verbose=False,
    )


def memory_keeper() -> Agent:
    profile = agent_profile("memory_keeper")
    return Agent(
        role=profile["role"],
        goal=profile["goal"],
        backstory=profile["backstory"],
        llm=planner_llm(),
        tools=[SaveEmployeeFactTool()],
        max_iter=2,
        max_execution_time=20,
        allow_delegation=False,
        allow_code_execution=False,
        verbose=False,
    )


def require_hr_agents() -> None:
    policy_researcher()
    answer_composer()
    memory_keeper()
    CalculateLeaveDaysTool()
    CountWorkingDaysTool()
    SearchConversationTool()
