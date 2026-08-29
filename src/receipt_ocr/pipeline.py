"""End-to-end receipt OCR pipeline orchestration."""

from pathlib import Path
from typing import Any


class ReceiptOCRPipeline:
    """Coordinate the replaceable stages of the receipt processing pipeline."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}

    def run(self, image_path: str | Path) -> dict[str, Any]:
        """Process one receipt image and return a structured record.

        Stage implementations and dependency wiring will be added after the
        initial OCR and extraction strategies have been selected.
        """
        raise NotImplementedError("Pipeline stages have not been implemented yet.")

