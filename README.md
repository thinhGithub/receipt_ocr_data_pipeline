# Receipt OCR Data Pipeline

## Overview

`receipt_ocr_data_pipeline` is a modular proof-of-concept data processing
pipeline that converts receipt images with different layouts into structured
records. OCR is treated as a replaceable component rather than the research
focus of the project.

## Problem Statement

Receipt images are unstructured and vary in layout, image quality, language,
and formatting. This project initially extracts and standardizes three fields:

- Store name
- Date
- Total amount

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

The pipeline is currently a skeleton. The intended entry point is:

```python
from receipt_ocr.pipeline import ReceiptOCRPipeline

pipeline = ReceiptOCRPipeline()
result = pipeline.run("data/raw/example.jpg")
```

Each processing stage will be implemented behind the corresponding module in
`src/receipt_ocr/`.

## Experiments

Experiments should compare controlled pipeline variants, including direct OCR
versus preprocessed OCR, and regex versus keyword or fuzzy field extraction.
Store experiment configuration separately and write generated artifacts to
`reports/` rather than embedding reusable logic in notebooks.

## Results

Evaluation will report field-level extraction quality and data quality changes
before and after cleaning, normalization, and validation. Generated metrics,
figures, and tables belong in their matching `reports/` subdirectories.

