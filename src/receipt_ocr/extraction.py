"""Intentionally simple rule-based receipt field extraction."""

from __future__ import annotations

import re

from receipt_ocr.normalization import find_timestamp, normalize_text


TOTAL_KEYWORDS = (
    "tong cong",
    "tong tien",
    "thanh toan",
    "phai tra",
    "total",
)
SELLER_EXCLUSIONS = (
    "hoa don",
    "dia chi",
    "address",
    "dien thoai",
    "ma so thue",
    "ngay",
    "receipt",
)
ADDRESS_KEYWORDS = (
    re.compile(r"\bdia chi\b"),
    re.compile(r"\baddress\b"),
    re.compile(r"\bdc\b"),
)
ADDRESS_HINTS = (
    re.compile(r"\bduong\b"),
    re.compile(r"\bphuong\b"),
    re.compile(r"\bhuyen\b"),
    re.compile(r"\bthanh pho\b"),
    re.compile(r"\btp\b"),
    re.compile(r"\bkhu\b"),
    re.compile(r"\bto\s+\d"),
    re.compile(r"\bch[og]\s+[a-z]"),
    re.compile(r"^so\s+\d"),
    re.compile(r"\bcam pha\b"),
    re.compile(r"\bquang ninh\b"),
    re.compile(r"\bqn\b"),
)
ADDRESS_STOPS = (
    "hoa don",
    "ngay",
    "thoi gian",
    "dien thoai",
    "hotline",
    "website",
    "ma so thue",
    "nhan vien",
    "thu ngan",
    "ten hang",
)


def extract_timestamp(lines: list[str]) -> str | None:
    for line in lines:
        found = find_timestamp(line)
        if found:
            return found[0]
    return None


def _address_score(line: str) -> int:
    normalized = normalize_text(line)
    if not normalized or any(stop in normalized for stop in ADDRESS_STOPS):
        return -1
    keyword_score = 4 * sum(pattern.search(normalized) is not None for pattern in ADDRESS_KEYWORDS)
    hint_score = sum(pattern.search(normalized) is not None for pattern in ADDRESS_HINTS)
    return keyword_score + hint_score


def extract_address(lines: list[str]) -> str | None:
    """Extract an address header and merge adjacent address-like lines."""
    header = lines[:10]
    scored = [(score, index) for index, line in enumerate(header) if (score := _address_score(line)) > 0]

    if scored:
        _, start = max(scored, key=lambda item: (item[0], -item[1]))
    else:
        seller = extract_seller(header)
        if seller is None:
            return None
        seller_index = next(index for index, line in enumerate(header) if line.strip() == seller)
        start = seller_index + 1
        if seller_index > 3 or start >= len(header) or _address_score(header[start]) < 0:
            return None

    selected = [header[start].strip()]
    for line in header[start + 1 : start + 3]:
        normalized = normalize_text(line)
        if not normalized or any(stop in normalized for stop in ADDRESS_STOPS):
            break
        if _address_score(line) <= 0:
            break
        selected.append(line.strip())
    return " ".join(selected) if selected else None


def extract_total_cost(lines: list[str]) -> str | None:
    amount_pattern = re.compile(
        r"(?<!\d)\d{1,3}(?:[.,\s]\d{3})+(?!\d)|(?<!\d)\d{4,9}(?!\d)"
    )
    candidates: list[tuple[int, int, str]] = []
    for index, line in enumerate(lines):
        normalized = normalize_text(line)
        keyword_score = sum(keyword in normalized for keyword in TOTAL_KEYWORDS)
        if not keyword_score:
            continue
        for match in amount_pattern.findall(line):
            candidates.append((keyword_score, index, match.strip()))
    if candidates:
        return max(candidates, key=lambda item: (item[0], item[1]))[2]
    return None


def extract_seller(lines: list[str]) -> str | None:
    """Use the first informative line; receipt headers usually contain seller."""
    for line in lines[:8]:
        normalized = normalize_text(line)
        if len(normalized) < 3 or not re.search(r"[a-z]", normalized):
            continue
        if any(keyword in normalized for keyword in SELLER_EXCLUSIONS):
            continue
        if sum(char.isdigit() for char in line) > len(line) * 0.4:
            continue
        return line.strip()
    return None


def extract_fields(lines: list[str]) -> dict[str, str | None]:
    return {
        "seller": extract_seller(lines),
        "address": extract_address(lines),
        "timestamp": extract_timestamp(lines),
        "total_cost": extract_total_cost(lines),
    }
