"""Smoke tests for pipeline orchestration."""

import pytest

from receipt_ocr.pipeline import ReceiptOCRPipeline


def test_pipeline_is_an_explicit_skeleton() -> None:
    pipeline = ReceiptOCRPipeline()

    with pytest.raises(NotImplementedError):
        pipeline.run("receipt.jpg")

