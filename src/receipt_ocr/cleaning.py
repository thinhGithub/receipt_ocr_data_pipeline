"""Raw OCR text cleaning components."""

from __future__ import annotations

import unicodedata


def clean_ocr_line(value: object) -> str:
    """Apply lossless Unicode and whitespace cleanup to one OCR line."""
    text = unicodedata.normalize("NFC", str(value or ""))
    return " ".join(text.split())


def clean_ocr_lines(lines: list[str]) -> list[str]:
    return [cleaned for line in lines if (cleaned := clean_ocr_line(line))]
