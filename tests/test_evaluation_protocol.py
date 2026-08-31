from pathlib import Path

import pandas as pd
import pytest

from receipt_ocr.evaluation_protocol import create_split_manifest, load_or_create_manifest
from receipt_ocr.ocr_cache import OCRCache, build_ocr_cache_key


def test_split_manifest_is_reproducible_and_disjoint() -> None:
    ids = [f"image-{index}.jpg" for index in range(20)]
    first = create_split_manifest(ids, seed=42, final_fraction=0.2)
    second = create_split_manifest(list(reversed(ids)), seed=42, final_fraction=0.2)
    pd.testing.assert_frame_equal(first, second)
    assert set(first["split"]) == {"development", "final"}
    assert (first["split"] == "final").sum() == 4


def test_existing_manifest_rejects_dataset_drift(tmp_path: Path) -> None:
    path = tmp_path / "manifest.csv"
    load_or_create_manifest(["a.jpg", "b.jpg"], path)
    with pytest.raises(ValueError, match="does not match"):
        load_or_create_manifest(["a.jpg", "c.jpg"], path)


def test_ocr_cache_reuses_result_and_invalidates_on_config(tmp_path: Path) -> None:
    image = tmp_path / "receipt.jpg"
    image.write_bytes(b"image contents")
    cache = OCRCache(tmp_path / "cache")
    calls = 0

    def runner() -> dict:
        nonlocal calls
        calls += 1
        return {"text": "OCR", "lines": [], "words": []}

    config = {"language": "vie+eng", "psm": 6, "preprocessing": {"enabled": False}}
    first, first_hit, first_key = cache.get_or_run(image, config, runner)
    second, second_hit, second_key = cache.get_or_run(image, config, runner)
    assert first == second
    assert first_hit is False
    assert second_hit is True
    assert first_key == second_key
    assert calls == 1

    changed = {**config, "psm": 4}
    assert build_ocr_cache_key(image, changed) != first_key
    cache.get_or_run(image, changed, runner)
    assert calls == 2
