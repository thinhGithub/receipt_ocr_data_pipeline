# Receipt OCR Data Pipeline

Pipeline POC chuyển ảnh hóa đơn tiếng Việt thành dữ liệu có cấu trúc gồm bốn
trường: `seller`, `address`, `timestamp` và `total_cost`.

## Luồng xử lý

```text
Ảnh hóa đơn -> Preprocessing (tùy chọn) -> Tesseract OCR -> Làm sạch văn bản
             -> Trích xuất trường -> Chuẩn hóa -> Kiểm tra hợp lệ
             -> JSON/JSONL
```

Hệ thống hiện hỗ trợ:

- Tesseract pretrained với ngôn ngữ `vie+eng`;
- preprocessing theo chuỗi xoay ảnh, crop hóa đơn và resize; có thêm các
  biến thể grayscale, CLAHE và perspective crop để thử nghiệm;
- trích xuất bốn trường bằng regex và keyword, gồm địa chỉ nhiều dòng;
- chuẩn hóa ngày giờ và số tiền;
- cache kết quả OCR theo nội dung ảnh và cấu hình;
- chia tập `development`/`final` cố định để đánh giá;
- lưu kết quả dạng JSONL khi bật trong cấu hình.

Preprocessing đã được tích hợp vào `ReceiptOCRPipeline`, nhưng mặc định vẫn
tắt (`preprocessing.enabled: false`) để việc bật nó là một quyết định tường minh.
Biến thể đang được đánh giá chính là `orientation_crop_resize`.

## Cài đặt

Yêu cầu Python 3.10+, Tesseract OCR và language data `vie`, `eng`.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
# source .venv/bin/activate

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

Tạo báo cáo HTML để xem ảnh, ground truth và prediction cạnh nhau:

```bash
python scripts/build_html_report.py
```

Mở `reports/metrics/ocr_baseline/development/report.html` bằng trình duyệt.

Baseline development hiện tại (40 ảnh):

| Trường | Coverage | Chỉ số chính |
|---|---:|---:|
| Seller | 94,87% | Fuzzy similarity: 44,95% |
| Address | 97,22% | Fuzzy similarity: 52,31% |
| Timestamp | 79,41% | Exact match: 67,65% |
| Total cost | 53,85% | Exact numeric match: 35,90% |
| Macro | 81,34% | Exact match: 30,43% |

### Step 1: orientation + resize

Chạy trên đúng 40 ảnh baseline, xoay khi Tesseract OSD đủ tin cậy và chuẩn hóa
chiều rộng ảnh về 1600 px:

```bash
python scripts/run_ocr_baseline.py \
  --split development \
  --sample-file reports/metrics/ocr_baseline/development/sample.csv \
  --preprocessing orientation_resize \
  --resize-width 1600 \
  --output-dir reports/metrics/ocr_step1_orientation_resize
```

Kết quả: macro coverage tăng từ 81,34% lên 82,57%; macro exact match tăng từ
30,43% lên 36,85%. Cả 40 ảnh được resize và không ảnh nào cần xoay.

### Step 2: thử nghiệm grayscale, CLAHE và crop

Step 2 kế thừa orientation/resize của Step 1, chuyển grayscale rồi tăng tương
phản cục bộ bằng CLAHE (`clip limit 2.0`, lưới `8x8`). Trên cùng 40 ảnh, macro
coverage đạt 84,60% nhưng macro exact match giảm còn 27,73%. Các thử
nghiệm crop sau resize và perspective crop cũng không vượt Step 1 về exact match.

### Step 3: orientation + crop + resize

Pipeline preprocessing hiện tại xoay ảnh trước, phát hiện vùng giấy theo contour
và màu, crop có padding, sau đó mới resize về chiều rộng 1600 px. Crop
chỉ được áp dụng khi diện tích và điểm tin cậy đủ ngưỡng; nếu không,
pipeline giữ ảnh đã xoay rồi resize.

```bash
python scripts/run_ocr_baseline.py \
  --split development \
  --sample-file reports/metrics/ocr_baseline/development/sample.csv \
  --preprocessing orientation_crop_resize \
  --preprocessed-dir data/interim/mc_ocr2021/preprocessed/step3_orientation_crop_resize \
  --resize-width 1600 \
  --crop-padding 0.01 \
  --min-receipt-area 0.08 \
  --min-crop-score 0.38 \
  --output-dir reports/metrics/ocr_step3_orientation_crop_resize
```

Kết quả development trên cùng 40 ảnh:

| Trường | Coverage | Chỉ số chính |
|---|---:|---:|
| Seller | 100,00% | Exact match: 20,51%; fuzzy similarity: 45,09% |
| Address | 100,00% | Exact match: 19,44%; fuzzy similarity: 61,86% |
| Timestamp | 82,35% | Exact match: 67,65% |
| Total cost | 64,10% | Exact numeric match: 35,90% |
| Macro | 86,61% | Exact match: 35,88% |

Step 3 cho coverage cao nhất trong các thử nghiệm hiện có, nhưng exact match
chưa vượt Step 1. Vì vậy preprocessing được xem là tạm ổn; nút thắt tiếp
theo là trích xuất trường từ OCR text.

## Dùng pipeline

```python
from receipt_ocr.pipeline import ReceiptOCRPipeline

result = ReceiptOCRPipeline().run("data/raw/example.jpg")
```

Bật preprocessing bằng cấu hình runtime:

```python
pipeline = ReceiptOCRPipeline(
    config={"preprocessing": {"enabled": True, "variant": "orientation_crop_resize"}}
)
result = pipeline.run("data/raw/example.jpg")
```

Muốn ghi vào `data/processed/receipts.jsonl`, đặt `storage.enabled: true` trong
`configs/default.yaml`.

## Kiểm thử

```bash
python -m pytest -q
```

## Việc tiếp theo

- Xây dựng error attribution: phân biệt lỗi OCR với lỗi extraction/normalization.
- Cải thiện `total_cost` extraction, đặc biệt khi hóa đơn có nhiều dòng
  `tổng tiền`, `giảm`, `khách trả` và `trả lại`.
- Cải thiện seller/address theo vị trí dòng và layout thay vì chỉ dùng keyword.
- Sau khi chốt extraction, thử nghiệm các Tesseract PSM nếu error attribution cho
  thấy OCR vẫn là nút thắt.
- Chạy tập final và bổ sung notebook báo cáo.
