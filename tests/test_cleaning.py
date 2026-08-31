"""Tests for OCR text cleaning rules."""

from receipt_ocr.cleaning import clean_ocr_line, clean_ocr_lines


def test_repairs_euro_sign_misread_as_c_before_text() -> None:
    assert clean_ocr_line("€ÔNG TY") == "CÔNG TY"
    assert clean_ocr_line("€   ong ty") == "C ong ty"


def test_preserves_euro_currency_amount() -> None:
    assert clean_ocr_line("Tổng: € 10") == "Tổng: € 10"


def test_normalizes_whitespace_and_removes_empty_lines() -> None:
    assert clean_ocr_lines(["  Cửa   hàng  ", "   ", "\tĐịa chỉ\n"]) == [
        "Cửa hàng",
        "Địa chỉ",
    ]
