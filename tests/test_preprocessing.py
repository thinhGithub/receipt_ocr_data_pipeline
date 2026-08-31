from pathlib import Path

from PIL import Image
import cv2

import receipt_ocr.preprocessing as preprocessing


def test_orientation_resize_rotates_and_resizes(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.jpg"
    output = tmp_path / "output.jpg"
    Image.new("RGB", (100, 200), "white").save(source)
    monkeypatch.setattr(
        preprocessing,
        "detect_orientation",
        lambda _: {
            "detected_rotation": 90,
            "orientation_confidence": 10.0,
            "osd_succeeded": True,
        },
    )

    metadata = preprocessing.orientation_resize(
        source, output, target_width=400, max_upscale=2.0, use_ocr_fallback=False
    )

    assert metadata["applied_rotation"] == 90
    assert metadata["original_width"] == 100
    assert metadata["original_height"] == 200
    assert metadata["output_width"] == 400
    assert metadata["output_height"] == 200
    with Image.open(output) as result:
        assert result.size == (400, 200)


def test_orientation_resize_ignores_low_confidence(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.jpg"
    output = tmp_path / "output.jpg"
    Image.new("RGB", (100, 200), "white").save(source)
    monkeypatch.setattr(
        preprocessing,
        "detect_orientation",
        lambda _: {
            "detected_rotation": 90,
            "orientation_confidence": 2.0,
            "osd_succeeded": True,
        },
    )

    metadata = preprocessing.orientation_resize(
        source, output, target_width=400, max_upscale=2.0, use_ocr_fallback=False
    )

    assert metadata["applied_rotation"] == 0
    with Image.open(output) as result:
        assert result.size == (200, 400)


def test_step2_outputs_grayscale_clahe_image(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.jpg"
    output = tmp_path / "output.jpg"
    Image.new("RGB", (100, 200), (180, 140, 100)).save(source)
    monkeypatch.setattr(
        preprocessing,
        "detect_orientation",
        lambda _: {
            "detected_rotation": 0,
            "orientation_confidence": 10.0,
            "osd_succeeded": True,
        },
    )

    metadata = preprocessing.orientation_resize_grayscale_clahe(
        source, output, target_width=200, max_upscale=2.0
    )

    result = cv2.imread(str(output), cv2.IMREAD_UNCHANGED)
    assert result is not None and result.ndim == 2
    assert result.shape == (400, 200)
    assert metadata["grayscale_applied"] is True
    assert metadata["contrast_method"] == "clahe"


def test_grayscale_variant_does_not_apply_contrast(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.jpg"
    output = tmp_path / "output.jpg"
    Image.new("RGB", (100, 200), (180, 140, 100)).save(source)
    monkeypatch.setattr(
        preprocessing,
        "detect_orientation",
        lambda _: {
            "detected_rotation": 0,
            "orientation_confidence": 10.0,
            "osd_succeeded": True,
        },
    )

    metadata = preprocessing.orientation_resize_grayscale(
        source, output, target_width=200, max_upscale=2.0
    )

    result = cv2.imread(str(output), cv2.IMREAD_UNCHANGED)
    assert result is not None and result.ndim == 2
    assert metadata["grayscale_applied"] is True
    assert metadata["contrast_method"] == "none"


def test_crop_variant_removes_background(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.jpg"
    output = tmp_path / "output.jpg"
    canvas = Image.new("RGB", (400, 500), (40, 40, 40))
    receipt = Image.new("RGB", (260, 400), "white")
    canvas.paste(receipt, (70, 50))
    canvas.save(source)
    monkeypatch.setattr(
        preprocessing,
        "detect_orientation",
        lambda _: {
            "detected_rotation": 0,
            "orientation_confidence": 10.0,
            "osd_succeeded": True,
        },
    )

    metadata = preprocessing.orientation_resize_crop(
        source, output, target_width=400, max_upscale=1.0, crop_padding=0.03
    )

    assert metadata["crop_applied"] is True
    assert metadata["crop_output_width"] == 400
    assert metadata["crop_output_height"] > metadata["crop_output_width"]


def test_canonical_processing3_pipeline_crops_then_resizes(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    output = tmp_path / "output.jpg"
    canvas = Image.new("RGB", (400, 500), (35, 35, 35))
    canvas.paste(Image.new("RGB", (260, 400), "white"), (70, 50))
    canvas.save(source)

    metadata = preprocessing.orientation_crop_resize(
        source,
        output,
        target_width=800,
        max_upscale=4.0,
        min_receipt_area=0.08,
        min_crop_score=0.30,
    )

    assert metadata["crop_applied"] is True
    assert metadata["output_width"] == 800
    assert metadata["resize_scale"] > 1.0
