"""Tests for end-to-end pipeline orchestration."""

import json
from pathlib import Path

import pytest

from receipt_ocr.pipeline import ReceiptOCRPipeline, load_config


def fake_ocr(image_path: str | Path, language: str, psm: int) -> dict:
    assert Path(image_path).is_file()
    assert language == "vie+eng"
    assert psm == 6
    texts = [
        "MINIMART ANAN",
        "Chợ Sủi Phú Thị Gia Lâm",
        "noise below threshold",
        "Ngày: 11/08/2020 08:06",
        "Tổng tiền: 74,000",
    ]
    return {
        "text": "\n".join(texts),
        "lines": [
            {"text": text, "mean_confidence": 10.0 if "noise" in text else 95.0}
            for text in texts
        ],
        "words": [],
    }


def test_load_default_config() -> None:
    config = load_config()
    assert config["ocr"]["language"] == "vie+eng"
    assert config["extraction"]["fields"] == [
        "seller", "address", "timestamp", "total_cost"
    ]


def test_pipeline_runs_end_to_end(tmp_path: Path) -> None:
    image = tmp_path / "receipt.jpg"
    image.write_bytes(b"test image placeholder")
    pipeline = ReceiptOCRPipeline(
        config={"ocr": {"cache": {"directory": str(tmp_path / "cache")}}},
        ocr_runner=fake_ocr,
        check_ocr=False,
    )

    result = pipeline.run(image)

    assert result["seller"] == "MINIMART ANAN"
    assert result["address"] == "Chợ Sủi Phú Thị Gia Lâm"
    assert result["timestamp"] == "2020-08-11 08:06"
    assert result["total_cost"] == 74000
    assert result["validation"] == {"is_valid": True, "errors": {}}
    assert result["pipeline"]["preprocessing_applied"] is False
    assert "noise below threshold" not in result["ocr"]["cleaned_lines"]
    assert result["pipeline"]["ocr_cache_hit"] is False

    cached = pipeline.run(image)
    assert cached["pipeline"]["ocr_cache_hit"] is True
    assert cached["pipeline"]["ocr_cache_key"] == result["pipeline"]["ocr_cache_key"]


def test_config_override_and_optional_storage(tmp_path: Path) -> None:
    image = tmp_path / "receipt.jpg"
    image.write_bytes(b"test image placeholder")
    output_dir = tmp_path / "output"
    pipeline = ReceiptOCRPipeline(
        config={
            "extraction": {"fields": ["timestamp", "total_cost"]},
            "ocr": {"cache": {"directory": str(tmp_path / "cache")}},
            "storage": {"enabled": True, "output_dir": str(output_dir)},
        },
        ocr_runner=fake_ocr,
        check_ocr=False,
    )

    result = pipeline.run(image)

    assert "seller" not in result
    assert result["timestamp"] == "2020-08-11 08:06"
    output_path = Path(result["output_path"])
    assert output_path.is_file()
    saved = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert saved["total_cost"] == 74000


def test_pipeline_rejects_missing_image() -> None:
    pipeline = ReceiptOCRPipeline(ocr_runner=fake_ocr, check_ocr=False)
    with pytest.raises(FileNotFoundError):
        pipeline.run("missing.jpg")


def test_pipeline_rejects_unsupported_engine() -> None:
    with pytest.raises(ValueError, match="tesseract"):
        ReceiptOCRPipeline(config={"ocr": {"engine": "unknown"}})
