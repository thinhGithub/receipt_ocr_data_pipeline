"""Small OCR adapters used by the baseline experiment."""

from __future__ import annotations

import csv
import io
import shutil
import subprocess
from pathlib import Path


def check_tesseract(language: str = "vie+eng") -> None:
    """Fail early with an actionable message when Tesseract is unavailable."""
    if shutil.which("tesseract") is None:
        raise RuntimeError(
            "Tesseract was not found on PATH. Install Tesseract OCR and the "
            "Vietnamese traineddata file, then rerun this command."
        )

    result = subprocess.run(
        ["tesseract", "--list-langs"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    installed = set(result.stdout.splitlines())
    missing = set(language.split("+")) - installed
    if missing:
        raise RuntimeError(
            "Missing Tesseract language data: " + ", ".join(sorted(missing))
        )


def run_tesseract(
    image_path: str | Path,
    language: str = "vie+eng",
    psm: int = 6,
) -> dict:
    """Run pretrained Tesseract and return text plus word-level evidence."""
    command = [
        "tesseract",
        str(image_path),
        "stdout",
        "-l",
        language,
        "--psm",
        str(psm),
        "tsv",
    ]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    words: list[dict] = []
    line_words: dict[tuple[int, int, int], list[str]] = {}
    line_confidences: dict[tuple[int, int, int], list[float]] = {}

    for row in csv.DictReader(io.StringIO(result.stdout), delimiter="\t"):
        text = (row.get("text") or "").strip()
        if not text:
            continue
        try:
            confidence = float(row["conf"])
        except (TypeError, ValueError):
            confidence = -1.0
        if confidence < 0:
            continue

        word = {
            "text": text,
            "confidence": confidence,
            "left": int(row["left"]),
            "top": int(row["top"]),
            "width": int(row["width"]),
            "height": int(row["height"]),
            "block_num": int(row["block_num"]),
            "par_num": int(row["par_num"]),
            "line_num": int(row["line_num"]),
        }
        words.append(word)
        key = (word["block_num"], word["par_num"], word["line_num"])
        line_words.setdefault(key, []).append(text)
        line_confidences.setdefault(key, []).append(confidence)

    lines = [
        {
            "text": " ".join(tokens),
            "mean_confidence": sum(line_confidences[key]) / len(tokens),
        }
        for key, tokens in line_words.items()
    ]
    return {
        "text": "\n".join(line["text"] for line in lines),
        "lines": lines,
        "words": words,
    }
