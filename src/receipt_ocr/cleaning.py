"""Raw OCR text cleaning components."""

from __future__ import annotations

import re
import unicodedata


# Tesseract can confuse an uppercase C with the euro sign in Vietnamese words.
# Keep actual currency values such as "€ 10" unchanged by requiring a letter next.
EURO_AS_C_PATTERN = re.compile(r"€(?=\s*[A-Za-zÀ-ỹ])")


def clean_ocr_line(value: object) -> str:
    """Normalize one OCR line and repair high-confidence character confusions."""
    text = unicodedata.normalize("NFC", str(value or ""))
    text = EURO_AS_C_PATTERN.sub("C", text)
    return " ".join(text.split())


def clean_ocr_lines(lines: list[str]) -> list[str]:
    return [cleaned for line in lines if (cleaned := clean_ocr_line(line))]
