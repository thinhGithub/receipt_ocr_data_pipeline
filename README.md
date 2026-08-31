# Receipt OCR Data Pipeline

## Overview

`receipt_ocr_data_pipeline` is a modular proof-of-concept data processing
pipeline that converts receipt images with different layouts into structured
records. OCR is treated as a replaceable component rather than the research
focus of the project.

## Problem Statement

Receipt images are unstructured and vary in layout, image quality, language,
and formatting. The pipeline extracts and normalizes four fields:

- Seller
- Address
- Timestamp
- Total cost

The goal is to make every processing stage independently testable and easy to
replace while keeping the POC simple.

## Pipeline

```text
Receipt Images
  -> Data Ingestion
  -> Image Quality Check / Preprocessing
  -> OCR
  -> Raw Text
  -> Text Cleaning
  -> Field Extraction
  -> Data Normalization
  -> Data Validation
  -> Structured Data Storage
  -> Evaluation
```

## Project Structure

```text
configs/                 Runtime and experiment configuration
data/                    Raw, external, interim, and processed data layers
docs/                    Design and project documentation
notebooks/               Exploration and experiment notebooks only
reports/                 Generated figures, tables, and metrics
scripts/                 Thin command-line entry scripts
src/receipt_ocr/         Reusable pipeline source code
tests/                   Automated tests
```

Reusable logic belongs in `src/receipt_ocr/`; notebooks should import that
logic instead of duplicating it.

## Dataset

Place original receipt images in `data/raw/`. Put third-party reference data
in `data/external/`, intermediate artifacts in `data/interim/`, and final
structured outputs in `data/processed/`. Dataset contents are ignored by Git
by default; only placeholder files are tracked.

Document dataset sources, licenses, annotation format, and split strategy here
once a dataset has been selected.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Activate the environment for your operating system.
python -m pip install -e ".[dev]"
```

Copy `.env.example` to `.env` if local environment variables are needed.
OCR engines and their Python adapters will be added after an engine is chosen.

## Usage

The end-to-end entry point loads `configs/default.yaml` and runs OCR, text
cleaning, field extraction, normalization, and validation:

```python
from receipt_ocr.pipeline import ReceiptOCRPipeline

pipeline = ReceiptOCRPipeline()
result = pipeline.run("data/raw/example.jpg")
```

Preprocessing and result storage are disabled by default. Storage can be enabled
with `storage.enabled: true`; preprocessing will be integrated in a subsequent
experiment step.

## Experiments

Experiments should compare controlled pipeline variants, including direct OCR
versus preprocessed OCR, and regex versus keyword or fuzzy field extraction.
Store experiment configuration separately and write generated artifacts to
`reports/` rather than embedding reusable logic in notebooks.

## Results

Evaluation will report field-level extraction quality and data quality changes
before and after cleaning, normalization, and validation. Generated metrics,
figures, and tables belong in their matching `reports/` subdirectories.

## OCR baseline POC

The baseline uses the pretrained Tesseract `vie+eng` language models and does
not train or fine-tune an OCR model. Install Tesseract OCR, include the `vie`
and `eng` traineddata files, and make sure `tesseract` is available on `PATH`.

Run a reproducible random sample of 40 receipts:

```bash
python scripts/run_ocr_baseline.py --sample-size 40 --seed 42
```

The first run creates a persistent `development`/`final` split manifest and a
content-addressed OCR cache. Re-running the command reuses OCR while allowing
the extraction and evaluation rules to change:

```bash
# Rule development only; this is the default split.
python scripts/run_ocr_baseline.py --split development --sample-size 40

# Final reporting after rules are frozen. A sample size of 0 means all images
# assigned to the final split.
python scripts/run_ocr_baseline.py --split final --sample-size 0
```

Use `--refresh-cache` only when OCR itself must be rerun. Cache keys include the
image contents, OCR engine/language/PSM, and preprocessing configuration, so a
future preprocessing variant cannot silently reuse incompatible OCR output.

Artifacts are written below `reports/metrics/ocr_baseline/<split>/`: `sample.csv`,
`raw_ocr.jsonl`, `results.csv`, `results_detailed.csv`, `metrics.json`, and
`error_examples.csv`. `experiment.json` records the manifest hash, OCR settings,
cache hit/miss counts, seed, and source paths.

Reuse the exact same sample after changing extraction rules:

```bash
python scripts/run_ocr_baseline.py \
  --sample-file reports/metrics/ocr_baseline/development/sample.csv
```
