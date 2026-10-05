import re
from datetime import date, timedelta

WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}
WEEKDAY_MENTION = re.compile(
    r"\b(?:(next|this)\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)
ISO_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


def upcoming_weekday(today: date, weekday: int) -> date:
    days_ahead = (weekday - today.weekday()) % 7
    if days_ahead == 0:
        return today + timedelta(days=7)
    return today + timedelta(days=days_ahead)


def mentioned_date(today: date, qualifier: str, weekday_name: str) -> date:
    weekday = WEEKDAYS[weekday_name.lower()]
    if qualifier.lower() == "this":
        days_ahead = (weekday - today.weekday()) % 7
        return today + timedelta(days=days_ahead)
    return upcoming_weekday(today, weekday)


def calendar_days_inclusive(start: date, end: date) -> int:
    return (end - start).days + 1


def count_working_days(from_date: date, to_date: date) -> int:
    if to_date <= from_date:
        return 0
    count = 0
    cursor = from_date + timedelta(days=1)
    while cursor < to_date:
        if cursor.weekday() < 5:
            count += 1
        cursor += timedelta(days=1)
    return count


def working_days_before(today: date, start: date) -> int:
    return count_working_days(today, start)


def date_pair(question: str, today: date) -> tuple[date, date] | None:
    parsed: list[date] = []
    for match in ISO_DATE.finditer(question):
        try:
            parsed.append(date.fromisoformat(match.group(1)))
        except ValueError:
            continue
    if len(parsed) >= 2:
        return parsed[0], parsed[1]
    mentions = list(WEEKDAY_MENTION.finditer(question))
    if len(mentions) < 2:
        return None
    start = mentioned_date(today, mentions[0].group(1) or "", mentions[0].group(2))
    end = mentioned_date(today, mentions[1].group(1) or "", mentions[1].group(2))
    if end < start:
        end = end + timedelta(days=7)
    return start, end


def describe_date_span(question: str, today: date) -> str:
    pair = date_pair(question, today)
    if pair is None:
        return ""
    start, end = pair
    if end < start:
        return ""
    calendar_days = calendar_days_inclusive(start, end)
    return "\n".join(
        [
            f"today: {today.isoformat()}",
            f"startDate: {start.isoformat()} ({start.strftime('%A %-d %B %Y')})",
            f"endDate: {end.isoformat()} ({end.strftime('%A %-d %B %Y')})",
            f"calendarDaysInclusive: {calendar_days}",
            f"workingDaysBeforeStart: {working_days_before(today, start)}",
        ]
    )
