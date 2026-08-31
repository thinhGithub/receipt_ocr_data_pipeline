"""Reproducible development/final split management."""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd


SPLITS = {"development", "final"}


def create_split_manifest(
    image_ids: list[str], seed: int = 42, final_fraction: float = 0.2
) -> pd.DataFrame:
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("img_id values must be unique before splitting.")
    if not 0 < final_fraction < 1:
        raise ValueError("final_fraction must be between 0 and 1.")
    shuffled = sorted(image_ids)
    random.Random(seed).shuffle(shuffled)
    final_count = max(1, round(len(shuffled) * final_fraction))
    final_ids = set(shuffled[:final_count])
    return pd.DataFrame(
        {
            "img_id": sorted(image_ids),
            "split": ["final" if image_id in final_ids else "development" for image_id in sorted(image_ids)],
        }
    )


def load_or_create_manifest(
    image_ids: list[str], manifest_path: str | Path, seed: int = 42,
    final_fraction: float = 0.2, rebuild: bool = False,
) -> pd.DataFrame:
    path = Path(manifest_path)
    if path.is_file() and not rebuild:
        manifest = pd.read_csv(path, encoding="utf-8-sig")
    else:
        manifest = create_split_manifest(image_ids, seed, final_fraction)
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(path, index=False, encoding="utf-8-sig")

    if set(manifest.columns) != {"img_id", "split"}:
        raise ValueError("Split manifest must contain exactly img_id and split columns.")
    if manifest["img_id"].duplicated().any() or set(manifest["img_id"]) != set(image_ids):
        raise ValueError("Split manifest does not match the current ground truth img_id set.")
    unknown = set(manifest["split"]) - SPLITS
    if unknown:
        raise ValueError(f"Unknown manifest splits: {', '.join(sorted(unknown))}")
    return manifest
