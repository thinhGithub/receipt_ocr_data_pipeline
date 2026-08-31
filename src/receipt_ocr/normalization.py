"""Normalization shared by extraction and evaluation."""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime


DATE_PATTERN = re.compile(
    r"(?<!\d)(\d{1,2})\s*([/.-])\s*(\d{1,2})\s*\2\s*(\d{2,4})(?!\d)"
)
TIME_PATTERN = re.compile(
    r"(?<!\d)(\d{1,2})\s*[:.]\s*(\d{2})(?:\s*[:.]\s*(\d{2}))?(?!\d)"
)
MAX_DATE_TIME_GAP = 12


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower().replace("đ", "d")
    return " ".join(re.findall(r"[a-z0-9]+", text))


def _valid_date(match: re.Match[str]) -> datetime | None:
    day, _, month, year = match.groups()
    normalized_year = int(f"20{year}" if len(year) == 2 else year)
    try:
        return datetime(normalized_year, int(month), int(day))
    except ValueError:
        return None


def _valid_time(match: re.Match[str]) -> tuple[int, int, int | None] | None:
    hour, minute, second = match.groups()
    parsed = int(hour), int(minute), int(second) if second is not None else None
    if parsed[0] > 23 or parsed[1] > 59 or (parsed[2] is not None and parsed[2] > 59):
        return None
    return parsed


def find_timestamp(value: object) -> tuple[str, str] | None:
    """Return the first valid timestamp as ``(source_text, normalized_text)``."""
    text = str(value or "")
    time_matches = [match for match in TIME_PATTERN.finditer(text) if _valid_time(match)]

    for date_match in DATE_PATTERN.finditer(text):
        parsed_date = _valid_date(date_match)
        if parsed_date is None:
            continue

        nearby_times = [
            match
            for match in time_matches
            if (
                0 <= match.start() - date_match.end() <= MAX_DATE_TIME_GAP
                or 0 <= date_match.start() - match.end() <= MAX_DATE_TIME_GAP
            )
        ]
        time_match = min(
            nearby_times,
            key=lambda match: min(
                abs(match.start() - date_match.end()),
                abs(date_match.start() - match.end()),
            ),
            default=None,
        )

        normalized = parsed_date.strftime("%Y-%m-%d")
        start, end = date_match.span()
        if time_match is not None:
            parsed_time = _valid_time(time_match)
            assert parsed_time is not None
            hour, minute, second = parsed_time
            normalized += f" {hour:02d}:{minute:02d}"
            if second is not None:
                normalized += f":{second:02d}"
            start = min(start, time_match.start())
            end = max(end, time_match.end())

        return text[start:end].strip(), normalized
    return None


def normalize_timestamp(value: object) -> str:
    found = find_timestamp(value)
    return found[1] if found else ""


def numeric_amount(value: object) -> str:
    """Return the last plausible monetary number as integer minor-free text."""
    text = str(value or "")
    candidates = re.findall(
        r"(?<!\d)\d{1,3}(?:[.,]\d{3})+(?:[.,]\d{2})?(?!\d)"
        r"|(?<!\d)\d{4,12}(?!\d)",
        text,
    )
    parsed: list[int] = []
    for candidate in candidates:
        # Receipts sometimes render a decimal suffix (e.g. 16,200.00).
        without_decimal = re.sub(r"[.,]00$", "", candidate)
        digits = re.sub(r"\D", "", without_decimal)
        if digits:
            parsed.append(int(digits))
    return str(parsed[-1]) if parsed else ""
