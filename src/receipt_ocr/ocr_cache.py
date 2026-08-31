"""Content-addressed cache for deterministic OCR reuse."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable


def build_ocr_signature(
    engine: str, language: str, psm: int, preprocessing: dict[str, Any]
) -> dict[str, Any]:
    preprocessing_signature = (
        preprocessing if preprocessing.get("enabled") else {"enabled": False}
    )
    return {
        "engine": engine,
        "language": language,
        "psm": psm,
        "preprocessing": preprocessing_signature,
    }


def build_ocr_cache_key(image_path: str | Path, config: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    with Path(image_path).open("rb") as image:
        for chunk in iter(lambda: image.read(1024 * 1024), b""):
            digest.update(chunk)
    digest.update(json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


class OCRCache:
    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)

    def path_for(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def get_or_run(
        self,
        image_path: str | Path,
        config: dict[str, Any],
        runner: Callable[[], dict[str, Any]],
        refresh: bool = False,
    ) -> tuple[dict[str, Any], bool, str]:
        key = build_ocr_cache_key(image_path, config)
        path = self.path_for(key)
        if path.is_file() and not refresh:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload["ocr"], True, key

        ocr = runner()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"cache_key": key, "config": config, "ocr": ocr}, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(path)
        return ocr, False, key
