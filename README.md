# Receipt OCR Data Pipeline

Pipeline POC chuyển ảnh hóa đơn tiếng Việt thành dữ liệu có cấu trúc gồm bốn
trường: `seller`, `address`, `timestamp` và `total_cost`.

## Luồng xử lý

```text
Ảnh hóa đơn -> Tesseract OCR -> Làm sạch văn bản -> Trích xuất trường
             -> Chuẩn hóa -> Kiểm tra hợp lệ -> JSON/JSONL
```

Hệ thống hiện hỗ trợ:

- Tesseract pretrained với ngôn ngữ `vie+eng`;
- trích xuất bốn trường bằng regex và keyword, gồm địa chỉ nhiều dòng;
- chuẩn hóa ngày giờ và số tiền;
- cache kết quả OCR theo nội dung ảnh và cấu hình;
- chia tập `development`/`final` cố định để đánh giá;
- lưu kết quả dạng JSONL khi bật trong cấu hình.

Preprocessing ảnh thực tế chưa được tích hợp; ảnh hiện được đưa trực tiếp vào
Tesseract.

## Cài đặt

Yêu cầu Python 3.10+, Tesseract OCR và language data `vie`, `eng`.

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
```

Cấu hình mặc định nằm tại `configs/default.yaml`.

## Chuẩn bị dữ liệu

Đặt bộ MC-OCR2021 trong `data/raw/mc_ocr2021/`, sau đó chạy:

```bash
python scripts/inspect_dataset.py
python scripts/build_ground_truth.py
```

Dữ liệu đã làm sạch có 1.153 ảnh hợp lệ. Ground truth được lưu trong
`data/interim/mc_ocr2021/`.

## Chạy baseline

```bash
# Phát triển rule trên mẫu 40 ảnh cố định
python scripts/run_ocr_baseline.py --split development --sample-size 40 --seed 42

# Đánh giá cuối cùng sau khi đã chốt rule
python scripts/run_ocr_baseline.py --split final --sample-size 0
```

Kết quả nằm trong `reports/metrics/ocr_baseline/<split>/`. OCR đã cache sẽ được
tái sử dụng; chỉ dùng `--refresh-cache` khi cần chạy lại OCR.

Baseline development hiện tại (40 ảnh):

| Trường | Coverage | Chỉ số chính |
|---|---:|---:|
| Seller | 94,87% | Fuzzy similarity: 44,95% |
| Address | 97,22% | Fuzzy similarity: 52,31% |
| Timestamp | 79,41% | Exact match: 67,65% |
| Total cost | 53,85% | Exact numeric match: 35,90% |
| Macro | 81,34% | Exact match: 30,43% |

## Dùng pipeline

```python
from receipt_ocr.pipeline import ReceiptOCRPipeline

result = ReceiptOCRPipeline().run("data/raw/example.jpg")
```

Muốn ghi vào `data/processed/receipts.jsonl`, đặt `storage.enabled: true` trong
`configs/default.yaml`.

## Kiểm thử

```bash
python -m pytest -q
```

## Việc tiếp theo

- Tích hợp preprocessing ảnh và so sánh với ảnh gốc.
- Thử nghiệm các chế độ Tesseract PSM.
- Cải thiện rule cho seller, address và total cost.
- Chạy tập final và bổ sung notebook báo cáo.
