import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import median

from fastapi import HTTPException

from app.activity.trace import StepDraft, TurnTrace
from app.core.database import open_app_database


@dataclass(frozen=True)
class ActivityRun:
    id: str
    organization_id: str
    organization_name: str
    question: str
    standalone_question: str
    route: str
    plan_json: str
    status: str
    total_ms: int
    first_token_ms: int | None
    created_at: str
    rating: str
    feedback_note: str


@dataclass(frozen=True)
class ActivityStep:
    name: str
    status: str
    input_summary: str
    output_summary: str
    started_ms: int
    ended_ms: int


@dataclass(frozen=True)
class RouteSpeed:
    route: str
    turn_count: int
    median_total_ms: int | None
    slowest_total_ms: int | None
    median_first_token_ms: int | None
    cache_hits: int
    total_target_ms: int | None
    first_token_target_ms: int | None
    baseline: str


def begin_activity(trace: TurnTrace) -> str:
    run_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    with open_app_database() as connection:
        connection.execute(
            """
            INSERT INTO agent_runs (
                id, session_id, organization_id, client_id, engine, question,
                standalone_question, route, plan_json, status, total_ms, first_token_ms, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                trace.session_id,
                trace.organization_id,
                trace.client_id,
                trace.engine,
                trace.question,
                trace.standalone_question,
                trace.route,
                trace.plan_json,
                "running",
                0,
                None,
                created_at,
            ),
        )
    trace.run_id = run_id
    return run_id


def update_activity_plan(trace: TurnTrace) -> None:
    if trace.run_id == "":
        return
    with open_app_database() as connection:
        connection.execute(
            """
            UPDATE agent_runs
            SET standalone_question = ?, route = ?, plan_json = ?, total_ms = ?
            WHERE id = ? AND status = 'running'
            """,
            (trace.standalone_question, trace.route, trace.plan_json, trace.elapsed_ms(), trace.run_id),
        )


def record_activity_step(trace: TurnTrace, step: StepDraft) -> None:
    if trace.run_id == "":
        return
    with open_app_database() as connection:
        connection.execute(
            """
            INSERT INTO agent_steps (
                id, run_id, kind, name, input_summary, output_summary, status, started_ms, ended_ms
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                trace.run_id,
                step.kind,
                step.name,
                step.input_summary,
                step.output_summary,
                step.status,
                step.started_ms,
                step.ended_ms,
            ),
        )
        connection.execute(
            """
            UPDATE agent_runs
            SET standalone_question = ?, route = ?, plan_json = ?, total_ms = ?
            WHERE id = ?
            """,
            (trace.standalone_question, trace.route, trace.plan_json, trace.elapsed_ms(), trace.run_id),
        )


def note_first_token(trace: TurnTrace) -> None:
    if trace.run_id == "" or trace.first_token_ms is not None:
        return
    trace.first_token_ms = trace.elapsed_ms()
    with open_app_database() as connection:
        connection.execute(
            """
            UPDATE agent_runs
            SET first_token_ms = ?,
                total_ms = CASE WHEN total_ms < ? THEN ? ELSE total_ms END
            WHERE id = ? AND first_token_ms IS NULL
            """,
            (trace.first_token_ms, trace.first_token_ms, trace.first_token_ms, trace.run_id),
        )


def finish_activity(trace: TurnTrace, status: str) -> None:
    if trace.run_id == "":
        return
    with open_app_database() as connection:
        connection.execute(
            """
            UPDATE agent_runs
            SET status = ?, total_ms = ?, standalone_question = ?, route = ?, plan_json = ?,
                first_token_ms = COALESCE(?, first_token_ms)
            WHERE id = ?
            """,
            (
                status,
                trace.elapsed_ms(),
                trace.standalone_question,
                trace.route,
                trace.plan_json,
                trace.first_token_ms,
                trace.run_id,
            ),
        )


def read_live_steps(client_id: str, site_host: str) -> tuple[bool, list[ActivityStep]]:
    with open_app_database() as connection:
        row = connection.execute(
            """
            SELECT agent_runs.id
            FROM agent_runs
            JOIN chat_sessions ON chat_sessions.id = agent_runs.session_id
            WHERE agent_runs.client_id = ?
              AND chat_sessions.site_host = ?
              AND agent_runs.status = 'running'
            ORDER BY agent_runs.created_at DESC, agent_runs.rowid DESC
            LIMIT 1
            """,
            (client_id, site_host),
        ).fetchone()
        if row is None:
            return False, []
        step_rows = connection.execute(
            """
            SELECT name, status, input_summary, output_summary, started_ms, ended_ms
            FROM agent_steps
            WHERE run_id = ?
            ORDER BY started_ms, rowid
            """,
            (str(row["id"]),),
        ).fetchall()
    return True, [
        ActivityStep(
            name=str(step["name"]),
            status=str(step["status"]),
            input_summary=str(step["input_summary"]),
            output_summary=str(step["output_summary"]),
            started_ms=int(step["started_ms"]),
            ended_ms=int(step["ended_ms"]),
        )
        for step in step_rows
    ]


def list_activity(
    limit: int = 50,
    rated_down: bool = False,
    organization_id: str = "",
) -> list[ActivityRun]:
    with open_app_database() as connection:
        rows = connection.execute(
            """
            SELECT
                agent_runs.id,
                agent_runs.organization_id,
                agent_runs.question,
                agent_runs.standalone_question,
                agent_runs.route,
                agent_runs.plan_json,
                agent_runs.status,
                agent_runs.total_ms,
                agent_runs.first_token_ms,
                agent_runs.created_at,
                organizations.name AS organization_name,
                COALESCE(answer_feedback.rating, '') AS rating,
                COALESCE(answer_feedback.note, '') AS feedback_note
            FROM agent_runs
            JOIN organizations ON organizations.id = agent_runs.organization_id
            LEFT JOIN answer_feedback ON answer_feedback.run_id = agent_runs.id
            WHERE (? = 0 OR answer_feedback.rating = 'down')
              AND (? = '' OR agent_runs.organization_id = ?)
            ORDER BY agent_runs.created_at DESC, agent_runs.rowid DESC
            LIMIT ?
            """,
            (1 if rated_down else 0, organization_id, organization_id, limit),
        ).fetchall()
    return [
        ActivityRun(
            id=str(row["id"]),
            organization_id=str(row["organization_id"]),
            organization_name=str(row["organization_name"]),
            question=str(row["question"]),
            standalone_question=str(row["standalone_question"]),
            route=str(row["route"]),
            plan_json=str(row["plan_json"]),
            status=str(row["status"]),
            total_ms=int(row["total_ms"]),
            first_token_ms=None if row["first_token_ms"] is None else int(row["first_token_ms"]),
            created_at=str(row["created_at"]),
            rating=str(row["rating"]),
            feedback_note=str(row["feedback_note"]),
        )
        for row in rows
    ]


def median_value(values: list[int]) -> int | None:
    return None if values == [] else int(median(values))


def route_targets(route: str) -> tuple[int | None, int | None, str]:
    if route == "ANSWER":
        return 35_000, 20_000, "Story 19: 40–100 s total"
    if route in {"GREETING", "RESET", "OUT_OF_SCOPE", "CLARIFY"}:
        return 3_000, None, "Story 19: 1–40 s total"
    if route == "pipeline":
        return None, None, "Story 19 pipeline baseline"
    return None, None, "No Story 19 baseline"


def speed_summary(limit: int = 50, organization_id: str = "") -> list[RouteSpeed]:
    completed = [
        run for run in list_activity(limit, organization_id=organization_id)
        if run.status in {"done", "cached"}
    ]
    grouped: dict[str, list[ActivityRun]] = {}
    for run in completed:
        grouped.setdefault(run.route, []).append(run)
    summary: list[RouteSpeed] = []
    for route in sorted(grouped):
        runs = grouped[route]
        totals = [run.total_ms for run in runs]
        first_tokens = [run.first_token_ms for run in runs if run.first_token_ms is not None]
        total_target, token_target, baseline = route_targets(route)
        summary.append(
            RouteSpeed(
                route=route,
                turn_count=len(runs),
                median_total_ms=median_value(totals),
                slowest_total_ms=max(totals) if totals else None,
                median_first_token_ms=median_value(first_tokens),
                cache_hits=sum(run.status == "cached" for run in runs),
                total_target_ms=total_target,
                first_token_target_ms=token_target,
                baseline=baseline,
            )
        )
    return summary


def read_activity(run_id: str) -> tuple[ActivityRun, list[ActivityStep]]:
    with open_app_database() as connection:
        row = connection.execute(
            """
            SELECT
                agent_runs.id,
                agent_runs.organization_id,
                agent_runs.question,
                agent_runs.standalone_question,
                agent_runs.route,
                agent_runs.plan_json,
                agent_runs.status,
                agent_runs.total_ms,
                agent_runs.first_token_ms,
                agent_runs.created_at,
                organizations.name AS organization_name,
                COALESCE(answer_feedback.rating, '') AS rating,
                COALESCE(answer_feedback.note, '') AS feedback_note
            FROM agent_runs
            JOIN organizations ON organizations.id = agent_runs.organization_id
            LEFT JOIN answer_feedback ON answer_feedback.run_id = agent_runs.id
            WHERE agent_runs.id = ?
            """,
            (run_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="That activity was not found.")
        step_rows = connection.execute(
            """
            SELECT name, status, input_summary, output_summary, started_ms, ended_ms
            FROM agent_steps
            WHERE run_id = ?
            ORDER BY started_ms, rowid
            """,
            (run_id,),
        ).fetchall()
    run = ActivityRun(
        id=str(row["id"]),
        organization_id=str(row["organization_id"]),
        organization_name=str(row["organization_name"]),
        question=str(row["question"]),
        standalone_question=str(row["standalone_question"]),
        route=str(row["route"]),
        plan_json=str(row["plan_json"]),
        status=str(row["status"]),
        total_ms=int(row["total_ms"]),
        first_token_ms=None if row["first_token_ms"] is None else int(row["first_token_ms"]),
        created_at=str(row["created_at"]),
        rating=str(row["rating"]),
        feedback_note=str(row["feedback_note"]),
    )
    steps = [
        ActivityStep(
            name=str(step["name"]),
            status=str(step["status"]),
            input_summary=str(step["input_summary"]),
            output_summary=str(step["output_summary"]),
            started_ms=int(step["started_ms"]),
            ended_ms=int(step["ended_ms"]),
        )
        for step in step_rows
    ]
    return run, steps
