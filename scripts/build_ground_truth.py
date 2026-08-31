from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd


EXPECTED_LABELS = [
    "SELLER",
    "ADDRESS",
    "TIMESTAMP",
    "TOTAL_COST",
]


def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def split_segments(value: object) -> list[str]:
    """
    Split MC-OCR annotation segments using the literal delimiter '|||'.

    Empty / missing values return an empty list.
    """
    if pd.isna(value):
        return []

    return [
        segment.strip()
        for segment in str(value).split("|||")
    ]


def build_ground_truth_row(row: pd.Series) -> dict:
    """
    Convert one MC-OCR annotation row into structured ground-truth fields.

    If multiple text segments belong to the same label, they are joined
    in their original order using a single space.

    Example:
        anno_texts:
            "Ngày: 21/05/2020|||20 : 42 : 52"

        anno_labels:
            "TIMESTAMP|||TIMESTAMP"

        output:
            timestamp = "Ngày: 21/05/2020 20 : 42 : 52"
    """
    texts = split_segments(row["anno_texts"])
    labels = split_segments(row["anno_labels"])

    if len(texts) != len(labels):
        raise ValueError(
            f"Annotation mismatch for img_id={row['img_id']}: "
            f"{len(texts)} text segments vs {len(labels)} labels"
        )

    grouped: dict[str, list[str]] = defaultdict(list)

    for text, label in zip(texts, labels):
        label = label.strip().upper()

        if not text:
            continue

        grouped[label].append(text)

    result = {
        "img_id": row["img_id"],
    }

    for label in EXPECTED_LABELS:
        column_name = label.lower()

        segments = grouped.get(label, [])

        result[column_name] = " ".join(segments) if segments else None
        result[f"{column_name}_segment_count"] = len(segments)

    # Keep useful source metadata for later analysis
    if "anno_num" in row.index:
        result["anno_num"] = row["anno_num"]

    if "anno_image_quality" in row.index:
        result["anno_image_quality"] = row["anno_image_quality"]

    # Detect unexpected labels instead of silently ignoring them
    unexpected_labels = sorted(
        set(grouped.keys()) - set(EXPECTED_LABELS)
    )

    result["unexpected_labels"] = (
        "|||".join(unexpected_labels)
        if unexpected_labels
        else None
    )

    return result


def build_ground_truth(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "img_id",
        "anno_texts",
        "anno_labels",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    records = []

    for _, row in df.iterrows():
        records.append(
            build_ground_truth_row(row)
        )

    return pd.DataFrame(records)


def validate_ground_truth(gt: pd.DataFrame) -> None:
    print_section("GROUND TRUTH VALIDATION")

    print(f"Rows: {len(gt):,}")

    print("\nMissing values by field:")

    for label in EXPECTED_LABELS:
        column = label.lower()

        missing_count = int(gt[column].isna().sum())
        missing_percent = gt[column].isna().mean() * 100

        print(
            f"{column:<12}: "
            f"{missing_count:>4} "
            f"({missing_percent:.2f}%)"
        )

    duplicate_ids = int(
        gt["img_id"].duplicated().sum()
    )

    print(f"\nDuplicated img_id: {duplicate_ids}")

    if "unexpected_labels" in gt.columns:
        unexpected = gt[
            gt["unexpected_labels"].notna()
        ]

        print(
            f"Rows with unexpected labels: "
            f"{len(unexpected)}"
        )

        if not unexpected.empty:
            print("\nUnexpected labels:")
            print(
                unexpected[
                    ["img_id", "unexpected_labels"]
                ].to_string(index=False)
            )


def print_label_statistics(gt: pd.DataFrame) -> None:
    print_section("FIELD STATISTICS")

    for label in EXPECTED_LABELS:
        column = label.lower()
        count_column = f"{column}_segment_count"

        available = int(
            gt[column].notna().sum()
        )

        total_segments = int(
            gt[count_column].sum()
        )

        mean_segments = float(
            gt[count_column].mean()
        )

        max_segments = int(
            gt[count_column].max()
        )

        print(f"\n{label}")
        print(f"  Receipts with field : {available:,}")
        print(f"  Total segments      : {total_segments:,}")
        print(f"  Mean segments/image : {mean_segments:.2f}")
        print(f"  Max segments/image  : {max_segments}")


def print_samples(
    gt: pd.DataFrame,
    n: int = 5,
) -> None:
    print_section("SAMPLE GROUND TRUTH")

    columns = [
        "img_id",
        "seller",
        "address",
        "timestamp",
        "total_cost",
    ]

    print(
        gt[columns]
        .head(n)
        .to_string(index=False)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build structured ground truth "
            "from cleaned MC-OCR 2021 annotations."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/interim/mc_ocr2021/"
            "annotations/mcocr_train_df.csv"
        ),
        help="Path to cleaned MC-OCR annotation CSV.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/interim/mc_ocr2021/"
            "ground_truth.csv"
        ),
        help="Output structured ground-truth CSV.",
    )

    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve()

    print_section("BUILD MC-OCR GROUND TRUTH")

    print(f"Input : {input_path}")
    print(f"Output: {output_path}")

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    df = pd.read_csv(
        input_path,
        encoding="utf-8-sig",
    )

    print(f"\nLoaded {len(df):,} annotation rows.")

    gt = build_ground_truth(df)

    validate_ground_truth(gt)
    print_label_statistics(gt)
    print_samples(gt)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # utf-8-sig makes Vietnamese display correctly
    # when opening the CSV directly in Excel.
    gt.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print_section("DONE")

    print(
        f"Ground truth saved successfully:\n"
        f"{output_path}"
    )


if __name__ == "__main__":
    main()