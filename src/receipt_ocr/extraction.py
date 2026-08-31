"""Intentionally simple rule-based receipt field extraction."""

from __future__ import annotations

import re

from receipt_ocr.normalization import find_timestamp, normalize_text


TOTAL_KEYWORD_SCORES = (
    ("tong tien phai t toan", 14),
    ("tong tien phai thanh toan", 14),
    ("tien thanh toan", 12),
    ("tong so thanh toan", 11),
    ("tong cong", 10),
    ("phai tra", 9),
    ("tong tien", 8),
    ("thanh toan", 7),
    ("total", 7),
)
TOTAL_NEGATIVE_KEYWORD_SCORES = (
    ("giam", 15),
    ("chiet khau", 15),
    ("khuyen mai", 15),
    ("tra lai", 13),
    ("tien thua", 13),
    ("khach tra", 5),
    ("tien mat", 5),
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
SELLER_ALIASES = (
    (re.compile(r"\bvincommerce\b"), "VinCommerce"),
    (re.compile(r"\bvin\s*mart\b"), "VinMart"),
    (re.compile(r"\bminimart\s+anan\b"), "MINIMART ANAN"),
    (re.compile(r"\bmilano\s+coffee\b"), "MILANO COFFEE"),
)
SELLER_HINTS = (
    "cong ty",
    "sieu thi",
    "cua hang",
    "nha sach",
    "coffee",
    "mart",
    "store",
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
    re.compile(r"\bqnh\b"),
    re.compile(r"\bp(?:\s*[.,:]\s*|\s+)[a-z0-9]"),
    re.compile(r"\bq(?:\s*[.,:]\s*|\s+)[a-z0-9]"),
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
    "mat hang",
    "quay",
    "phone",
    "wifi",
    "pass wifi",
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
    """Extract an address anchor and merge adjacent address-like header lines."""
    header = lines[:10]
    seller = extract_seller(header)
    seller_index = next(
        (index for index, line in enumerate(header) if line.strip() == seller), None
    )
    scored = [
        (score, index)
        for index, line in enumerate(header)
        if index != seller_index and (score := _address_score(line)) > 0
    ]

    if scored:
        _, anchor = max(scored, key=lambda item: (item[0], -item[1]))
    else:
        if seller_index is None:
            return None
        anchor = seller_index + 1
        if seller_index > 3 or anchor >= len(header) or _address_score(header[anchor]) < 0:
            return None

    start = anchor
    for index in range(anchor - 1, max(-1, anchor - 3), -1):
        if index == seller_index or _address_score(header[index]) <= 0:
            break
        start = index

    selected = []
    for index, line in enumerate(header[start : anchor + 3], start=start):
        if index == seller_index:
            continue
        normalized = normalize_text(line)
        if not normalized or any(stop in normalized for stop in ADDRESS_STOPS):
            break
        if index > anchor and _address_score(line) <= 0:
            break
        cleaned = line.strip()
        for alias_pattern, _ in SELLER_ALIASES:
            cleaned = re.sub(
                rf"^\s*{alias_pattern.pattern}\s*[-:|,]*\s*", "", cleaned, flags=re.IGNORECASE
            )
        if cleaned:
            selected.append(cleaned)
    return " ".join(selected) if selected else None


def extract_total_cost(lines: list[str]) -> str | None:
    amount_pattern = re.compile(
        r"(?<!\d)\d{1,3}(?:[.,\s]\d{3})+(?!\d)|(?<!\d)\d{4,9}(?!\d)"
    )
    candidates: list[tuple[int, int, str]] = []
    for index, line in enumerate(lines):
        normalized = normalize_text(line)
        positive_scores = [
            score for keyword, score in TOTAL_KEYWORD_SCORES if keyword in normalized
        ]
        if not positive_scores:
            continue
        keyword_score = max(positive_scores) - sum(
            score
            for keyword, score in TOTAL_NEGATIVE_KEYWORD_SCORES
            if keyword in normalized
        )
        for match in amount_pattern.findall(line):
            candidates.append((keyword_score, index, match.strip()))
    if candidates:
        return max(candidates, key=lambda item: (item[0], -item[1]))[2]
    return None


def extract_seller(lines: list[str]) -> str | None:
    """Score receipt header lines and return the most plausible seller."""
    header = lines[:8]
    for line in header:
        normalized = normalize_text(line)
        for pattern, canonical in SELLER_ALIASES:
            if pattern.search(normalized):
                return canonical

    candidates: list[tuple[float, int, str]] = []
    for index, line in enumerate(header):
        normalized = normalize_text(line)
        if len(normalized) < 3 or not re.search(r"[a-z]", normalized):
            continue
        if any(keyword in normalized for keyword in SELLER_EXCLUSIONS):
            continue
        if sum(char.isdigit() for char in line) > len(line) * 0.4:
            continue
        visible = [char for char in line if not char.isspace()]
        letter_ratio = sum(char.isalpha() for char in visible) / max(len(visible), 1)
        if letter_ratio < 0.45:
            continue
        hint_score = 3 * sum(hint in normalized for hint in SELLER_HINTS)
        address_penalty = 1.5 * max(_address_score(line), 0)
        score = 5 * letter_ratio + hint_score - address_penalty - 0.25 * index
        candidates.append((score, -index, line.strip()))
    return max(candidates)[2] if candidates else None


def extract_fields(lines: list[str]) -> dict[str, str | None]:
    return {
        "seller": extract_seller(lines),
        "address": extract_address(lines),
        "timestamp": extract_timestamp(lines),
        "total_cost": extract_total_cost(lines),
    }
