from receipt_ocr.evaluation import evaluate_results
from receipt_ocr.extraction import extract_fields
from receipt_ocr.normalization import numeric_amount, normalize_text, normalize_timestamp

import pandas as pd


def test_normalization() -> None:
    assert normalize_text("TỔNG TIỀN") == "tong tien"
    assert normalize_timestamp("Ngày: 21/05/2020 20 : 42 : 52") == "2020-05-21 20:42:52"
    assert numeric_amount("Tổng cộng 72,000 VNĐ") == "72000"
    assert numeric_amount("Tiền thanh toán 19 000") == "19000"


def test_timestamp_with_time_before_date() -> None:
    assert normalize_timestamp("Thời gian 09:05:45 - 15/08/2020") == "2020-08-15 09:05:45"


def test_timestamp_rejects_invalid_date_and_time() -> None:
    assert normalize_timestamp("40/08/2020") == ""
    assert normalize_timestamp("31/02/2020") == ""
    assert normalize_timestamp("15/08/2020 25:61") == "2020-08-15"


def test_extract_timestamp_skips_invalid_candidate() -> None:
    fields = extract_fields([
        "MINIMART ANAN",
        "OCR noise 40/08/2020",
        "Thời gian: 08:48:12 - 13/08/2020",
    ])
    assert fields["timestamp"] == "08:48:12 - 13/08/2020"


def test_extract_fields() -> None:
    fields = extract_fields([
        "MINIMART ANAN",
        "Địa chỉ: Chợ Sủi",
        "Ngày: 11/08/2020 08:06",
        "Tổng tiền: 74,000",
    ])
    assert fields == {
        "seller": "MINIMART ANAN",
        "address": "Địa chỉ: Chợ Sủi",
        "timestamp": "11/08/2020 08:06",
        "total_cost": "74,000",
    }


def test_extract_multiline_address() -> None:
    fields = extract_fields([
        "SCTC CO THO 104 TRAN PHU",
        "104 Trần Phú - phường Cẩm Tây - Thành phố Cẩm",
        "Phả - Quảng Ninh",
        "Hotline: 0963133818",
        "HÓA ĐƠN THANH TOÁN",
    ])
    assert fields["address"] == (
        "104 Trần Phú - phường Cẩm Tây - Thành phố Cẩm Phả - Quảng Ninh"
    )


def test_seller_skips_noisy_header_and_uses_known_alias() -> None:
    fields = extract_fields([
        "i ee, _.Ắ_Ắ. .Ă.__ớ tt ]",
        "VM+ QNH Dự án KDC lấn biển cọc 6",
        "VinCommerce TP, Cẩm Phả, T. Quảng Ninh",
    ])
    assert fields["seller"] == "VinCommerce"


def test_address_merges_lines_before_anchor_and_strips_seller() -> None:
    fields = extract_fields([
        "VinCommerce",
        "VM+ QNH Dự án KDC lấn biển cọc 6",
        "DA khu DCLB cọc 6, P. Cẩm Sơn",
        "VinCommerce TP. Cẩm Phả, T. Quảng Ninh",
        "024.71066866",
        "HÓA ĐƠN BÁN HÀNG",
    ])
    assert fields["address"] == (
        "VM+ QNH Dự án KDC lấn biển cọc 6 DA khu DCLB cọc 6, P. Cẩm Sơn "
        "TP. Cẩm Phả, T. Quảng Ninh"
    )


def test_total_prefers_payable_amount_over_discount() -> None:
    fields = extract_fields([
        "TỔNG TIỀN PHẢI T.TOÁN 8.435",
        "TỔNG TIỀN ĐÃ GIẢM -4.541",
        "TIỀN KHÁCH TRẢ 8.435",
    ])
    assert fields["total_cost"] == "8.435"


def test_total_prefers_grand_total_when_later_total_is_ocr_error() -> None:
    fields = extract_fields([
        "Tổng cộng: 222,000",
        "Tổng tiền: 222,100",
    ])
    assert fields["total_cost"] == "222,000"


def test_evaluation() -> None:
    frame = pd.DataFrame([{
        "seller_gt": "MINIMART ANAN", "seller_pred": "Minimart Anan",
        "address_gt": "Chợ Sủi, Phú Thị, Gia Lâm", "address_pred": "Cho Sui Phu Thi Gia Lam",
        "timestamp_gt": "Ngày 11/08/2020 08:06", "timestamp_pred": "11-08-2020 08:06",
        "total_cost_gt": "Tổng tiền 74,000", "total_cost_pred": "74000",
    }])
    metrics = evaluate_results(frame)
    assert metrics["seller"]["normalized_exact_match"] == 1.0
    assert metrics["address"]["normalized_exact_match"] == 1.0
    assert metrics["timestamp"]["normalized_exact_match"] == 1.0
    assert metrics["total_cost"]["exact_numeric_match"] == 1.0
    assert metrics["macro"]["prediction_coverage"] == 1.0
    assert metrics["macro"]["normalized_exact_match"] == 1.0


def test_evaluation_excludes_invalid_timestamp_ground_truth() -> None:
    frame = pd.DataFrame([
        {
            "seller_gt": None,
            "seller_pred": None,
            "address_gt": None,
            "address_pred": None,
            "timestamp_gt": "Ngày:",
            "timestamp_pred": "11/08/2020",
            "total_cost_gt": None,
            "total_cost_pred": None,
        }
    ])
    metrics = evaluate_results(frame)
    assert metrics["timestamp"]["gt_available"] == 0
    assert metrics["timestamp"]["normalized_exact_match"] == 0.0


def test_address_evaluation_counts_missing_prediction_as_zero() -> None:
    frame = pd.DataFrame([
        {
            "seller_gt": None,
            "seller_pred": None,
            "address_gt": "Chợ Sủi Phú Thị Gia Lâm",
            "address_pred": None,
            "timestamp_gt": None,
            "timestamp_pred": None,
            "total_cost_gt": None,
            "total_cost_pred": None,
        }
    ])
    metrics = evaluate_results(frame)
    assert metrics["address"]["prediction_coverage"] == 0.0
    assert metrics["address"]["mean_fuzzy_similarity"] == 0.0
