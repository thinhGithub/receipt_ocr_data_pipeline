from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from receipt_ocr.evaluation import evaluate_results, similarity
from receipt_ocr.evaluation_protocol import load_or_create_manifest
from receipt_ocr.cleaning import clean_ocr_lines
from receipt_ocr.extraction import extract_fields
from receipt_ocr.normalization import numeric_amount, normalize_timestamp
from receipt_ocr.ocr import check_tesseract, run_tesseract
from receipt_ocr.ocr_cache import OCRCache, build_ocr_signature
from receipt_ocr.preprocessing import (
    orientation_resize,
    orientation_crop_resize,
    orientation_resize_crop,
    orientation_resize_perspective_crop,
    orientation_resize_grayscale,
    orientation_resize_grayscale_clahe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small pretrained OCR baseline.")
    parser.add_argument("--ground-truth", type=Path, default=Path("data/interim/mc_ocr2021/ground_truth.csv"))
    parser.add_argument("--images-dir", type=Path, default=Path("data/raw/mc_ocr2021/train_images"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/metrics/ocr_baseline"))
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=["development", "final"], default="development")
    parser.add_argument("--manifest", type=Path, default=Path("data/interim/mc_ocr2021/split_manifest.csv"))
    parser.add_argument("--final-fraction", type=float, default=0.2)
    parser.add_argument("--rebuild-manifest", action="store_true")
    parser.add_argument("--language", default="vie+eng")
    parser.add_argument("--psm", type=int, default=6)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/interim/mc_ocr2021/ocr_cache"))
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--sample-file", type=Path, help="Reuse a CSV containing img_id values.")
    parser.add_argument("--preprocessing", choices=["none", "orientation_resize", "orientation_crop_resize", "orientation_resize_crop", "orientation_resize_perspective_crop", "orientation_resize_grayscale", "orientation_resize_grayscale_clahe"], default="none")
    parser.add_argument("--preprocessed-dir", type=Path, default=Path("data/interim/mc_ocr2021/preprocessed/step1_orientation_resize"))
    parser.add_argument("--resize-width", type=int, default=1600)
    parser.add_argument("--resize-max-upscale", type=float, default=4.0)
    parser.add_argument("--orientation-min-confidence", type=float, default=5.0)
    parser.add_argument("--clahe-clip-limit", type=float, default=2.0)
    parser.add_argument("--clahe-grid-size", type=int, default=8)
    parser.add_argument("--crop-padding", type=float, default=0.03)
    parser.add_argument("--crop-debug-dir", type=Path)
    parser.add_argument("--min-receipt-area", type=float, default=0.08)
    parser.add_argument("--min-crop-score", type=float, default=0.38)
    return parser.parse_args()


def select_sample(gt: pd.DataFrame, size: int, seed: int, sample_file: Path | None) -> pd.DataFrame:
    if sample_file:
        ids = pd.read_csv(sample_file)["img_id"]
        sample = gt[gt["img_id"].isin(ids)].copy()
        if len(sample) != len(ids):
            raise ValueError("Some img_id values in --sample-file are absent from ground truth.")
        return sample
    if size == 0:
        return gt.reset_index(drop=True)
    if not 1 <= size <= len(gt):
        raise ValueError(f"--sample-size must be 0 (all) or between 1 and {len(gt)}")
    indices = random.Random(seed).sample(range(len(gt)), size)
    return gt.iloc[indices].reset_index(drop=True)


def classify_errors(row: pd.Series) -> str:
    if float(row.get("anno_image_quality", 1.0)) < 0.5:
        return "low_image_quality"
    if not row["ocr_text"].strip():
        return "ocr_empty"
    if row.get("address_similarity", 1.0) < 0.8:
        return "ocr_or_address_extraction"
    if not row["timestamp_pred"] and row.get("timestamp_gt"):
        return "ocr_or_timestamp_extraction"
    if not row["total_cost_pred"] and row.get("total_cost_gt"):
        return "ocr_or_total_extraction"
    return "ocr_extraction_or_normalization"


def main() -> None:
    args = parse_args()
    check_tesseract(args.language)

    gt = pd.read_csv(args.ground_truth, encoding="utf-8-sig")
    manifest = load_or_create_manifest(
        gt["img_id"].tolist(), args.manifest, args.seed, args.final_fraction,
        rebuild=args.rebuild_manifest,
    )
    split_ids = manifest.loc[manifest["split"] == args.split, "img_id"]
    pool = gt[gt["img_id"].isin(split_ids)].copy()
    sample = select_sample(pool, args.sample_size, args.seed, args.sample_file)
    output_dir = args.output_dir / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    sample[["img_id"]].to_csv(output_dir / "sample.csv", index=False, encoding="utf-8-sig")

    raw_path = output_dir / "raw_ocr.jsonl"
    cache = OCRCache(args.cache_dir)
    preprocessing_config = {
        "enabled": args.preprocessing != "none",
        "variant": args.preprocessing,
        "orientation_strategy": "osd_then_landscape_ocr_score_v2",
        "resize_width": args.resize_width,
        "resize_max_upscale": args.resize_max_upscale,
        "orientation_min_confidence": args.orientation_min_confidence,
        "clahe_clip_limit": args.clahe_clip_limit,
        "clahe_grid_size": args.clahe_grid_size,
        "crop_padding": args.crop_padding,
        "min_receipt_area": args.min_receipt_area,
        "min_crop_score": args.min_crop_score,
        "crop_strategy": "four_point_perspective_v1" if args.preprocessing == "orientation_resize_perspective_crop" else "contour_paper_color_v2",
    }
    cache_signature = build_ocr_signature("tesseract", args.language, args.psm, preprocessing_config)
    cache_hits = 0
    records: list[dict] = []
    with raw_path.open("w", encoding="utf-8") as raw_file:
        for position, (_, gt_row) in enumerate(sample.iterrows(), start=1):
            image_path = args.images_dir / gt_row["img_id"]
            if not image_path.is_file():
                raise FileNotFoundError(f"Image not found: {image_path}")
            preprocess_metadata = {}
            ocr_image_path = image_path
            if args.preprocessing in {"orientation_resize", "orientation_crop_resize", "orientation_resize_crop", "orientation_resize_perspective_crop", "orientation_resize_grayscale", "orientation_resize_grayscale_clahe"}:
                ocr_image_path = args.preprocessed_dir / gt_row["img_id"]
                preprocess = {
                    "orientation_resize": orientation_resize,
                    "orientation_crop_resize": orientation_crop_resize,
                    "orientation_resize_crop": orientation_resize_crop,
                    "orientation_resize_perspective_crop": orientation_resize_perspective_crop,
                    "orientation_resize_grayscale": orientation_resize_grayscale,
                    "orientation_resize_grayscale_clahe": orientation_resize_grayscale_clahe,
                }[args.preprocessing]
                extra_options = (
                    {"clahe_clip_limit": args.clahe_clip_limit, "clahe_grid_size": args.clahe_grid_size}
                    if args.preprocessing == "orientation_resize_grayscale_clahe"
                    else {}
                )
                if args.preprocessing == "orientation_resize_crop":
                    extra_options = {"crop_padding": args.crop_padding}
                if args.preprocessing == "orientation_resize_perspective_crop":
                    extra_options = {
                        "crop_padding": args.crop_padding,
                        "debug_path": args.crop_debug_dir / gt_row["img_id"] if args.crop_debug_dir else None,
                    }
                if args.preprocessing == "orientation_crop_resize":
                    extra_options = {
                        "crop_padding": args.crop_padding,
                        "min_receipt_area": args.min_receipt_area,
                        "min_crop_score": args.min_crop_score,
                        "debug_path": args.crop_debug_dir / gt_row["img_id"] if args.crop_debug_dir else None,
                    }
                preprocess_metadata = preprocess(
                    image_path,
                    ocr_image_path,
                    target_width=args.resize_width,
                    max_upscale=args.resize_max_upscale,
                    min_orientation_confidence=args.orientation_min_confidence,
                    **extra_options,
                )
            ocr, cache_hit, cache_key = cache.get_or_run(
                ocr_image_path,
                cache_signature,
                lambda: run_tesseract(ocr_image_path, language=args.language, psm=args.psm),
                refresh=args.refresh_cache,
            )
            cache_hits += int(cache_hit)
            raw_file.write(json.dumps({"img_id": gt_row["img_id"], "cache_key": cache_key, **ocr}, ensure_ascii=False) + "\n")
            cleaned_lines = clean_ocr_lines([line["text"] for line in ocr["lines"]])
            fields = extract_fields(cleaned_lines)
            records.append({
                "img_id": gt_row["img_id"],
                "seller_gt": gt_row.get("seller"), "seller_pred": fields["seller"],
                "address_gt": gt_row.get("address"), "address_pred": fields["address"],
                "timestamp_gt": gt_row.get("timestamp"), "timestamp_pred": fields["timestamp"],
                "total_cost_gt": gt_row.get("total_cost"), "total_cost_pred": fields["total_cost"],
                "anno_image_quality": gt_row.get("anno_image_quality"),
                "ocr_text": ocr["text"],
                **preprocess_metadata,
            })
            source = "cache" if cache_hit else "ocr"
            print(f"[{position:02d}/{len(sample):02d}] [{source}] {gt_row['img_id']}")

    detailed = pd.DataFrame(records)
    detailed["seller_similarity"] = [similarity(a, b) for a, b in zip(detailed.seller_gt, detailed.seller_pred)]
    detailed["address_similarity"] = [similarity(a, b) for a, b in zip(detailed.address_gt, detailed.address_pred)]
    detailed["timestamp_match"] = [normalize_timestamp(a) != "" and normalize_timestamp(a) == normalize_timestamp(b) for a, b in zip(detailed.timestamp_gt, detailed.timestamp_pred)]
    detailed["total_cost_match"] = [numeric_amount(a) != "" and numeric_amount(a) == numeric_amount(b) for a, b in zip(detailed.total_cost_gt, detailed.total_cost_pred)]

    result_columns = ["img_id", "seller_gt", "seller_pred", "address_gt", "address_pred", "timestamp_gt", "timestamp_pred", "total_cost_gt", "total_cost_pred"]
    detailed[result_columns].to_csv(output_dir / "results.csv", index=False, encoding="utf-8-sig")
    detailed.to_csv(output_dir / "results_detailed.csv", index=False, encoding="utf-8-sig")

    metrics = evaluate_results(detailed)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    errors = detailed[(detailed["seller_similarity"] < 0.8) | (detailed["address_similarity"] < 0.8) | ~detailed["timestamp_match"] | ~detailed["total_cost_match"]].copy()
    errors["likely_error_source"] = errors.apply(classify_errors, axis=1)
    errors.sort_values(["anno_image_quality", "seller_similarity"]).head(20).to_csv(
        output_dir / "error_examples.csv", index=False, encoding="utf-8-sig"
    )
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    experiment = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "sample_size": len(sample),
        "seed": args.seed,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": manifest_hash,
        "ground_truth": str(args.ground_truth.resolve()),
        "ocr": cache_signature,
        "cache": {
            "directory": str(args.cache_dir.resolve()),
            "hits": cache_hits,
            "misses": len(sample) - cache_hits,
            "refresh": args.refresh_cache,
        },
    }
    (output_dir / "experiment.json").write_text(
        json.dumps(experiment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"OCR cache: {cache_hits} hit(s), {len(sample) - cache_hits} miss(es)")
    print(f"Artifacts saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
