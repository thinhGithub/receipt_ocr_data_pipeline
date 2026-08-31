"""End-to-end receipt OCR pipeline orchestration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import yaml

from receipt_ocr.cleaning import clean_ocr_lines
from receipt_ocr.extraction import extract_fields
from receipt_ocr.normalization import numeric_amount, normalize_timestamp
from receipt_ocr.ocr import check_tesseract, run_tesseract
from receipt_ocr.ocr_cache import OCRCache, build_ocr_signature
from receipt_ocr.storage import append_jsonl
from receipt_ocr.validation import validate_structured_fields


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
OCRRunner = Callable[[str | Path, str, int], dict[str, Any]]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError("Pipeline config must be a YAML mapping.")
    return config


class ReceiptOCRPipeline:
    """Run OCR, cleaning, KIE, normalization, validation, and optional storage."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        ocr_runner: OCRRunner = run_tesseract,
        check_ocr: bool = True,
    ) -> None:
        self.config = _deep_merge(load_config(config_path), config or {})
        self.ocr_runner = ocr_runner
        self.check_ocr = check_ocr
        self._validate_config()

    def _validate_config(self) -> None:
        ocr = self.config.get("ocr", {})
        if ocr.get("engine") != "tesseract":
            raise ValueError("Only the 'tesseract' OCR engine is currently supported.")
        fields = self.config.get("extraction", {}).get("fields", [])
        supported = {"seller", "address", "timestamp", "total_cost"}
        unknown = set(fields) - supported
        if unknown:
            raise ValueError(f"Unsupported extraction fields: {', '.join(sorted(unknown))}")
        storage = self.config.get("storage", {})
        if storage.get("enabled") and storage.get("format") != "jsonl":
            raise ValueError("Only JSONL storage is currently supported.")

    def _select_lines(self, ocr_result: dict[str, Any]) -> list[str]:
        threshold = float(self.config["ocr"].get("confidence_threshold", 0.0))
        threshold = threshold * 100 if 0 <= threshold <= 1 else threshold
        lines = [
            line["text"]
            for line in ocr_result.get("lines", [])
            if float(line.get("mean_confidence", 100.0)) >= threshold
        ]
        return clean_ocr_lines(lines)

    @staticmethod
    def _normalize_fields(raw_fields: dict[str, str | None]) -> dict[str, Any]:
        amount = numeric_amount(raw_fields.get("total_cost"))
        return {
            "seller": raw_fields.get("seller"),
            "address": raw_fields.get("address"),
            "timestamp": normalize_timestamp(raw_fields.get("timestamp")) or None,
            "total_cost": int(amount) if amount else None,
        }

    def run(self, image_path: str | Path) -> dict[str, Any]:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Receipt image not found: {path}")

        ocr_config = self.config["ocr"]
        language = str(ocr_config.get("language", "vie+eng"))
        psm = int(ocr_config.get("page_segmentation_mode", 6))
        if self.check_ocr and self.ocr_runner is run_tesseract:
            check_tesseract(language)

        # Step 4 will replace this direct pass-through with real preprocessing.
        cache_config = ocr_config.get("cache", {})
        cache_signature = build_ocr_signature(
            ocr_config["engine"], language, psm, self.config.get("preprocessing", {})
        )
        if cache_config.get("enabled"):
            cache = OCRCache(cache_config["directory"])
            ocr_result, cache_hit, cache_key = cache.get_or_run(
                path,
                cache_signature,
                lambda: self.ocr_runner(path, language, psm),
            )
        else:
            ocr_result = self.ocr_runner(path, language, psm)
            cache_hit, cache_key = False, None
        cleaned_lines = self._select_lines(ocr_result)
        all_raw_fields = extract_fields(cleaned_lines)
        configured_fields = self.config["extraction"]["fields"]
        raw_fields = {field: all_raw_fields[field] for field in configured_fields}
        all_normalized = self._normalize_fields(all_raw_fields)
        fields = {field: all_normalized[field] for field in configured_fields}

        result: dict[str, Any] = {
            "img_id": path.name,
            **fields,
            "raw_fields": raw_fields,
            "validation": validate_structured_fields(fields),
            "pipeline": {
                "preprocessing_applied": False,
                "ocr_engine": ocr_config["engine"],
                "ocr_language": language,
                "ocr_psm": psm,
                "ocr_cache_hit": cache_hit,
                "ocr_cache_key": cache_key,
            },
        }
        if self.config["storage"].get("save_intermediate_results"):
            result["ocr"] = {**ocr_result, "cleaned_lines": cleaned_lines}

        storage = self.config["storage"]
        if storage.get("enabled"):
            output = Path(storage["output_dir"]) / "receipts.jsonl"
            result["output_path"] = str(append_jsonl(result, output).resolve())
        return result
