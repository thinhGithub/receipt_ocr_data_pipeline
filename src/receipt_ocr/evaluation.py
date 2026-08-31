"""Field-level metrics for the OCR baseline."""

from __future__ import annotations

from difflib import SequenceMatcher

import pandas as pd

from receipt_ocr.normalization import numeric_amount, normalize_text, normalize_timestamp


def similarity(left: object, right: object) -> float:
    if pd.isna(left) or pd.isna(right):
        return 0.0
    return SequenceMatcher(None, normalize_text(left), normalize_text(right)).ratio()


def evaluate_results(results: pd.DataFrame) -> dict:
    seller_scored = results[results["seller_gt"].notna()].copy()
    address_scored = results[results["address_gt"].notna()].copy()
    timestamp_scored = results[
        results["timestamp_gt"].apply(normalize_timestamp).ne("")
    ].copy()
    total_scored = results[results["total_cost_gt"].notna()].copy()

    seller_scores = [
        similarity(gt, pred)
        for gt, pred in zip(seller_scored["seller_gt"], seller_scored["seller_pred"])
    ]
    address_scores = [
        similarity(gt, pred)
        for gt, pred in zip(address_scored["address_gt"], address_scored["address_pred"])
    ]
    timestamp_matches = [
        normalize_timestamp(gt) != "" and normalize_timestamp(gt) == normalize_timestamp(pred)
        for gt, pred in zip(timestamp_scored["timestamp_gt"], timestamp_scored["timestamp_pred"])
    ]
    total_matches = [
        numeric_amount(gt) != "" and numeric_amount(gt) == numeric_amount(pred)
        for gt, pred in zip(total_scored["total_cost_gt"], total_scored["total_cost_pred"])
    ]

    def coverage(frame: pd.DataFrame, column: str) -> float:
        return float(frame[column].notna().mean()) if len(frame) else 0.0

    field_metrics = {
        "seller": {
            "gt_available": int(len(seller_scored)),
            "prediction_coverage": coverage(seller_scored, "seller_pred"),
            "mean_fuzzy_similarity": sum(seller_scores) / len(seller_scores) if seller_scores else 0.0,
            "normalized_exact_match": sum(score == 1.0 for score in seller_scores) / len(seller_scores) if seller_scores else 0.0,
        },
        "address": {
            "gt_available": int(len(address_scored)),
            "prediction_coverage": coverage(address_scored, "address_pred"),
            "mean_fuzzy_similarity": sum(address_scores) / len(address_scores) if address_scores else 0.0,
            "normalized_exact_match": sum(score == 1.0 for score in address_scores) / len(address_scores) if address_scores else 0.0,
        },
        "timestamp": {
            "gt_available": int(len(timestamp_scored)),
            "prediction_coverage": coverage(timestamp_scored, "timestamp_pred"),
            "normalized_exact_match": sum(timestamp_matches) / len(timestamp_matches) if timestamp_matches else 0.0,
        },
        "total_cost": {
            "gt_available": int(len(total_scored)),
            "prediction_coverage": coverage(total_scored, "total_cost_pred"),
            "exact_numeric_match": sum(total_matches) / len(total_matches) if total_matches else 0.0,
        },
    }
    exact_scores = [
        field_metrics["seller"]["normalized_exact_match"],
        field_metrics["address"]["normalized_exact_match"],
        field_metrics["timestamp"]["normalized_exact_match"],
        field_metrics["total_cost"]["exact_numeric_match"],
    ]
    coverages = [field_metrics[field]["prediction_coverage"] for field in field_metrics]
    return {
        "sample_size": int(len(results)),
        **field_metrics,
        "macro": {
            "prediction_coverage": sum(coverages) / len(coverages),
            "normalized_exact_match": sum(exact_scores) / len(exact_scores),
        },
    }
