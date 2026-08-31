from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


# Các pattern thường xuất hiện khi UTF-8 bị decode sai
MOJIBAKE_PATTERNS = [
    "Ã",
    "Æ",
    "áº",
    "á»",
    "Ä",
    "ðŸ",
]


def load_csv(csv_path: Path) -> pd.DataFrame:
    """
    Thử đọc CSV với một số encoding phổ biến.
    Ưu tiên UTF-8 vì không nên vội kết luận file raw bị lỗi encoding.
    """
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1"]

    last_error = None

    for encoding in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            print(f"[OK] CSV loaded with encoding: {encoding}")
            return df
        except UnicodeDecodeError as exc:
            last_error = exc

    raise RuntimeError(
        f"Không thể đọc CSV bằng các encoding {encodings}"
    ) from last_error


def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def inspect_basic_info(df: pd.DataFrame) -> dict:
    print_section("1. BASIC DATASET INFORMATION")

    print(f"Rows       : {len(df):,}")
    print(f"Columns    : {len(df.columns)}")
    print(f"Column list: {list(df.columns)}")

    print("\nData types:")
    print(df.dtypes)

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": list(df.columns),
    }


def inspect_missing_values(df: pd.DataFrame) -> dict:
    print_section("2. MISSING VALUES")

    missing = df.isna().sum()
    missing_percent = df.isna().mean() * 100

    result = pd.DataFrame(
        {
            "missing_count": missing,
            "missing_percent": missing_percent.round(2),
        }
    )

    print(result)

    return {
        column: {
            "missing_count": int(result.loc[column, "missing_count"]),
            "missing_percent": float(result.loc[column, "missing_percent"]),
        }
        for column in result.index
    }


def inspect_duplicates(df: pd.DataFrame) -> dict:
    print_section("3. DUPLICATES")

    duplicated_rows = int(df.duplicated().sum())

    print(f"Duplicated full rows: {duplicated_rows}")

    duplicated_img_ids = None

    if "img_id" in df.columns:
        duplicated_img_ids = int(df["img_id"].duplicated().sum())
        print(f"Duplicated img_id   : {duplicated_img_ids}")

        if duplicated_img_ids:
            print("\nDuplicated image IDs:")
            print(
                df.loc[
                    df["img_id"].duplicated(keep=False),
                    "img_id",
                ].value_counts()
            )

    return {
        "duplicated_rows": duplicated_rows,
        "duplicated_img_ids": duplicated_img_ids,
    }


def inspect_annotation_counts(df: pd.DataFrame) -> dict:
    """
    anno_texts và anno_labels của MC-OCR thường được ngăn bằng |||.
    Kiểm tra số segment text có tương ứng với số label và anno_num hay không.
    """
    print_section("4. ANNOTATION CONSISTENCY")

    required = {"anno_texts", "anno_labels"}

    if not required.issubset(df.columns):
        print(
            "Không tìm thấy đủ anno_texts và anno_labels. "
            "Bỏ qua bước này."
        )
        return {}

    working_df = df.copy()

    working_df["text_segment_count"] = (
        working_df["anno_texts"]
        .fillna("")
        .astype(str)
        .apply(
            lambda x: len(x.split("|||"))
            if x.strip()
            else 0
        )
    )

    working_df["label_segment_count"] = (
        working_df["anno_labels"]
        .fillna("")
        .astype(str)
        .apply(
            lambda x: len(x.split("|||"))
            if x.strip()
            else 0
        )
    )

    mismatch_text_label = working_df[
        working_df["text_segment_count"]
        != working_df["label_segment_count"]
    ]

    print(
        "Rows where number of text segments != number of labels:",
        len(mismatch_text_label),
    )

    anno_num_mismatch = pd.DataFrame()

    if "anno_num" in working_df.columns:
        numeric_anno_num = pd.to_numeric(
            working_df["anno_num"],
            errors="coerce",
        )

        anno_num_mismatch = working_df[
            numeric_anno_num.notna()
            & (
                numeric_anno_num
                != working_df["text_segment_count"]
            )
        ]

        print(
            "Rows where anno_num != text segment count:",
            len(anno_num_mismatch),
        )

    print("\nText segment count statistics:")
    print(working_df["text_segment_count"].describe())

    return {
        "text_label_count_mismatch": int(len(mismatch_text_label)),
        "anno_num_mismatch": int(len(anno_num_mismatch)),
        "text_segments_mean": float(
            working_df["text_segment_count"].mean()
        ),
        "text_segments_min": int(
            working_df["text_segment_count"].min()
        ),
        "text_segments_max": int(
            working_df["text_segment_count"].max()
        ),
    }


def contains_mojibake(value: object) -> bool:
    if pd.isna(value):
        return False

    text = str(value)

    return any(
        pattern in text
        for pattern in MOJIBAKE_PATTERNS
    )


def inspect_encoding(
    df: pd.DataFrame,
    output_dir: Path,
) -> dict:
    print_section("5. POSSIBLE ENCODING / MOJIBAKE ISSUES")

    text_columns = [
        column
        for column in ["anno_texts", "anno_labels"]
        if column in df.columns
    ]

    if not text_columns:
        print("Không có text column phù hợp để kiểm tra.")
        return {}

    suspicious_mask = pd.Series(False, index=df.index)

    for column in text_columns:
        suspicious_mask |= df[column].apply(contains_mojibake)

    suspicious = df[suspicious_mask]

    print(f"Suspicious rows: {len(suspicious):,}")

    if not suspicious.empty:
        output_path = output_dir / "possible_mojibake_rows.csv"

        suspicious.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",
        )

        print(f"Saved suspicious rows -> {output_path}")

        if "anno_texts" in suspicious.columns:
            print("\nExamples:")

            for value in suspicious["anno_texts"].head(5):
                print("-" * 50)
                print(value)

    return {
        "possible_mojibake_rows": int(len(suspicious))
    }


def inspect_image_quality(df: pd.DataFrame) -> dict:
    print_section("6. IMAGE QUALITY")

    if "anno_image_quality" not in df.columns:
        print("Không tìm thấy cột anno_image_quality.")
        return {}

    quality = pd.to_numeric(
        df["anno_image_quality"],
        errors="coerce",
    )

    print(quality.describe())

    print("\nQuality thresholds:")
    print(f"< 0.3 : {(quality < 0.3).sum()}")
    print(f"< 0.5 : {(quality < 0.5).sum()}")
    print(f"< 0.7 : {(quality < 0.7).sum()}")

    return {
        "mean": float(quality.mean()),
        "median": float(quality.median()),
        "min": float(quality.min()),
        "max": float(quality.max()),
        "below_0_3": int((quality < 0.3).sum()),
        "below_0_5": int((quality < 0.5).sum()),
        "below_0_7": int((quality < 0.7).sum()),
    }


def inspect_labels(df: pd.DataFrame) -> dict:
    print_section("7. LABEL DISTRIBUTION")

    if "anno_labels" not in df.columns:
        print("Không tìm thấy anno_labels.")
        return {}

    labels = (
        df["anno_labels"]
        .dropna()
        .astype(str)
        .str.split("|||", regex=False)
        .explode()
        .str.strip()
    )

    labels = labels[labels != ""]

    counts = labels.value_counts()

    print(counts)

    return {
        str(label): int(count)
        for label, count in counts.items()
    }


def inspect_image_files(
    df: pd.DataFrame,
    images_dir: Path,
    check_readability: bool = False,
) -> dict:
    print_section("8. IMAGE FILE CONSISTENCY")

    if "img_id" not in df.columns:
        print("CSV không có cột img_id.")
        return {}

    if not images_dir.exists():
        print(f"Images directory not found: {images_dir}")
        return {}

    csv_images = set(
        df["img_id"]
        .dropna()
        .astype(str)
    )

    disk_files = {
        file.name
        for file in images_dir.iterdir()
        if file.is_file()
    }

    missing_images = csv_images - disk_files
    orphan_images = disk_files - csv_images

    print(f"Image IDs in CSV : {len(csv_images):,}")
    print(f"Files on disk    : {len(disk_files):,}")
    print(f"Missing images   : {len(missing_images):,}")
    print(f"Unreferenced     : {len(orphan_images):,}")

    if missing_images:
        print("\nExamples of missing images:")
        for name in sorted(missing_images)[:10]:
            print(f"  - {name}")

    if orphan_images:
        print("\nExamples of images not referenced by CSV:")
        for name in sorted(orphan_images)[:10]:
            print(f"  - {name}")

    corrupted = []

    if check_readability:
        print("\nChecking image readability...")

        try:
            from PIL import Image
        except ImportError:
            print(
                "Pillow chưa được cài. "
                "Chạy: pip install pillow"
            )
        else:
            for index, file in enumerate(
                images_dir.iterdir(),
                start=1,
            ):
                if not file.is_file():
                    continue

                try:
                    with Image.open(file) as image:
                        image.verify()
                except Exception:
                    corrupted.append(file.name)

                if index % 100 == 0:
                    print(f"Checked {index} images...")

            print(f"Unreadable/corrupted images: {len(corrupted)}")

    return {
        "csv_image_count": int(len(csv_images)),
        "disk_image_count": int(len(disk_files)),
        "missing_images": int(len(missing_images)),
        "unreferenced_images": int(len(orphan_images)),
        "corrupted_images": int(len(corrupted)),
    }


def inspect_sample(df: pd.DataFrame) -> None:
    print_section("9. SAMPLE RECORD")

    if df.empty:
        print("Dataset empty.")
        return

    row = df.iloc[0]

    for column in df.columns:
        print(f"\n[{column}]")
        print(row[column])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect MC-OCR 2021 dataset quality."
    )

    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(
            "data/raw/mc_ocr2021/annotations/mcocr_train_df.csv"
        ),
        help="Path to metadata CSV.",
    )

    parser.add_argument(
        "--images",
        type=Path,
        default=Path(
            "data/raw/mc_ocr2021/train_images"
        ),
        help="Path to receipt image directory.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "reports/metrics/dataset_inspection"
        ),
        help="Directory for inspection outputs.",
    )

    parser.add_argument(
        "--check-images",
        action="store_true",
        help="Open every image and verify that it is readable.",
    )

    args = parser.parse_args()

    csv_path = args.csv.resolve()
    images_dir = args.images.resolve()
    output_dir = args.output.resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"CSV    : {csv_path}")
    print(f"Images : {images_dir}")
    print(f"Output : {output_dir}")

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found: {csv_path}"
        )

    df = load_csv(csv_path)

    report = {}

    report["basic"] = inspect_basic_info(df)
    report["missing"] = inspect_missing_values(df)
    report["duplicates"] = inspect_duplicates(df)
    report["annotations"] = inspect_annotation_counts(df)
    report["encoding"] = inspect_encoding(
        df,
        output_dir,
    )
    report["image_quality"] = inspect_image_quality(df)
    report["labels"] = inspect_labels(df)
    report["images"] = inspect_image_files(
        df,
        images_dir,
        check_readability=args.check_images,
    )

    inspect_sample(df)

    report_path = output_dir / "inspection_report.json"

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print_section("DONE")
    print(f"Report saved to:\n{report_path}")


if __name__ == "__main__":
    main()