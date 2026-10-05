from app.core.settings import load_settings
from app.llm.chat_model import ChatModelFailed, generate_json
from app.rag.prompts import EXPANSION_PROMPT, EXPANSION_SCHEMA


def rewrite_queries(question: str, count: int) -> list[str]:
    try:
        parsed = generate_json(
            EXPANSION_PROMPT.format(count=count, question=question),
            EXPANSION_SCHEMA,
            60,
            load_settings().planner_model,
        )
    except ChatModelFailed:
        return []
    queries = parsed.get("queries", [])
    if not isinstance(queries, list):
        return []
    return [str(query).strip() for query in queries if str(query).strip() != ""]


def expand_queries(question: str) -> list[str]:
    count = load_settings().rag_expansion_count
    if count <= 0:
        return [question]
    seen: set[str] = set()
    queries: list[str] = []
    for candidate in [question, *rewrite_queries(question, count)]:
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        queries.append(candidate)
        if len(queries) > count:
            break
    return queries
